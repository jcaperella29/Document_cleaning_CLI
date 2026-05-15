import os
import shutil
import argparse
import uuid

import cv2
import h5py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pytesseract

from docclean.manifest import build_manifest, write_manifest
from docclean.engines.sbb import run_sbb_binarization


OCR_CONFIG = "--oem 3 --psm 6"


def configure_tesseract():
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return env_path

    system_path = shutil.which("tesseract")
    if system_path:
        pytesseract.pytesseract.tesseract_cmd = system_path
        return system_path

    windows_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    linux_default = "/usr/bin/tesseract"

    if os.path.exists(windows_default):
        pytesseract.pytesseract.tesseract_cmd = windows_default
        return windows_default

    if os.path.exists(linux_default):
        pytesseract.pytesseract.tesseract_cmd = linux_default
        return linux_default

    raise RuntimeError(
        "Tesseract is not installed or it's not in your PATH. "
        "Set TESSERACT_CMD or install Tesseract."
    )


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64

        layers = [
            nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False)
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


def validate_weights_folder(weights_folder):
    if not os.path.isdir(weights_folder):
        raise FileNotFoundError(f"Weights folder not found: {weights_folder}")

    mat_files = sorted(f for f in os.listdir(weights_folder) if f.endswith(".mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat weight files found in: {weights_folder}")

    return mat_files


def validate_input_folder(input_folder):
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    image_files = sorted(
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not image_files:
        raise FileNotFoundError(f"No valid image files found in: {input_folder}")

    return image_files


def load_h5_weights(mat_file_path, model):
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Weight file not found: {mat_file_path}")

    with h5py.File(mat_file_path, "r") as f:
        weights_datasets = []
        biases_datasets = []

        for key in f.keys():
            if key.startswith("#refs#"):
                for subkey in f[key]:
                    obj = f[f"{key}/{subkey}"]
                    if isinstance(obj, h5py.Dataset):
                        if len(obj.shape) == 4:
                            weights_datasets.append(obj)
                        elif len(obj.shape) == 2:
                            biases_datasets.append(obj)

        with torch.no_grad():
            weight_idx = 0
            bias_idx = 0

            for layer in model.dncnn:
                if isinstance(layer, nn.Conv2d):
                    if weight_idx < len(weights_datasets):
                        weight_data = weights_datasets[weight_idx][()]
                        if weight_data.shape == tuple(layer.weight.shape):
                            layer.weight.copy_(torch.tensor(weight_data))
                        weight_idx += 1

                    if layer.bias is not None and bias_idx < len(biases_datasets):
                        bias_data = biases_datasets[bias_idx][()]
                        if bias_data.shape == tuple(layer.bias.shape):
                            layer.bias.copy_(torch.tensor(bias_data))
                        bias_idx += 1


def denoise_with_cnn(model, noisy_image_path):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    if noisy_image is None:
        raise FileNotFoundError(
            f"Noisy image not found or unreadable: {noisy_image_path}"
        )

    noisy_image_float = noisy_image.astype(np.float32) / 255.0
    noisy_tensor = torch.from_numpy(noisy_image_float).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze(0).squeeze(0).cpu().numpy()

    denoised_tensor = np.clip(denoised_tensor, 0.0, 1.0)
    return (denoised_tensor * 255.0).round().astype(np.uint8)


def _safe_conf_values(data_dict):
    confs = []

    for value in data_dict.get("conf", []):
        try:
            conf = float(value)
        except (TypeError, ValueError):
            continue

        if conf >= 0:
            confs.append(conf)

    return confs


def ocr_text_quality(image, profile="ocr"):
    """
    Score an image using OCR signal.

    human:
        favors readable/crisp text without being too harsh.

    ocr:
        prioritizes OCR confidence and penalizes low-confidence junk tokens.
    """
    data = pytesseract.image_to_data(
        image,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    words = [text.strip() for text in data.get("text", []) if text and text.strip()]
    text_len = sum(len(word) for word in words)

    confs = _safe_conf_values(data)
    mean_conf = float(np.mean(confs)) if confs else 0.0

    laplacian_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    edge_score = min(laplacian_var / 1000.0, 10.0)

    std_score = float(np.std(image)) / 255.0
    low_conf_count = sum(1 for conf in confs if conf < 60)

    if profile == "human":
        return (
            (text_len * 0.8)
            + (mean_conf * 1.0)
            + (edge_score * 2.0)
            - (low_conf_count * 1.0)
        )

    return (
        (mean_conf * 5.0)
        + (len(words) * 0.75)
        + (edge_score * 0.5)
        + (std_score * 2.0)
        - (low_conf_count * 6.0)
    )

def ocr_metrics(image):
    """
    Return concrete OCR metrics for manifest.json and before/after comparisons.
    """
    data = pytesseract.image_to_data(
        image,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    low_confidence_tokens = 0
    confs = _safe_conf_values(data)

    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = text.strip() if text else ""
        if not text:
            continue

        words.append(text)

        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue

        if 0 <= conf_value < 60:
            low_confidence_tokens += 1

    mean_confidence = float(np.mean(confs)) if confs else 0.0

    return {
        "mean_confidence": round(mean_confidence, 3),
        "extracted_words": len(words),
        "extracted_characters": sum(len(word) for word in words),
        "low_confidence_tokens": low_confidence_tokens,
        "text_preview": " ".join(words[:50]),
    }


def compare_ocr_metrics(before_metrics, after_metrics):
    """
    Build delta metrics and a simple OCR-improvement interpretation.
    """
    delta_mean_confidence = round(
        after_metrics["mean_confidence"] - before_metrics["mean_confidence"],
        3,
    )
    delta_extracted_words = (
        after_metrics["extracted_words"] - before_metrics["extracted_words"]
    )
    delta_extracted_characters = (
        after_metrics["extracted_characters"]
        - before_metrics["extracted_characters"]
    )
    delta_low_confidence_tokens = (
        after_metrics["low_confidence_tokens"]
        - before_metrics["low_confidence_tokens"]
    )

    ocr_improved = (
        delta_mean_confidence > 0
        and delta_extracted_words >= 0
    )

    if ocr_improved and delta_low_confidence_tokens <= 0:
        ocr_quality_note = (
            "OCR confidence improved and low-confidence token count did not increase."
        )
    elif ocr_improved:
        ocr_quality_note = (
            "OCR confidence and extracted word count improved, "
            "but low-confidence token count also increased."
        )
    else:
        ocr_quality_note = (
            "OCR metrics did not clearly improve. "
            "Review output before using as OCR-optimized result."
        )

    return {
        "delta": {
            "mean_confidence": delta_mean_confidence,
            "extracted_words": delta_extracted_words,
            "extracted_characters": delta_extracted_characters,
            "low_confidence_tokens": delta_low_confidence_tokens,
        },
        "ocr_improved": ocr_improved,
        "recommended_for_ocr": ocr_improved,
        "ocr_quality_note": ocr_quality_note,
    }


def score_ocr_candidate(comparison):
    """
    Score a candidate using OCR deltas.

    Preference order:
    1. Prefer candidates that clearly improve OCR.
    2. Penalize outputs that destroy extracted text.
    3. Reward confidence, words, and characters.
    4. Penalize added low-confidence tokens.
    """
    delta = comparison["delta"]

    confidence_delta = delta["mean_confidence"]
    word_delta = delta["extracted_words"]
    character_delta = delta["extracted_characters"]
    low_conf_delta = delta["low_confidence_tokens"]

    score = (
        (confidence_delta * 5.0)
        + (word_delta * 2.0)
        + (character_delta * 0.15)
        - max(low_conf_delta, 0) * 1.5
    )

    if comparison["ocr_improved"]:
        score += 25.0
    else:
        score -= 25.0

    # Heavy penalty for destructive OCR collapse.
    if word_delta < 0:
        score += word_delta * 6.0

    if character_delta < 0:
        score += character_delta * 0.45

    # Reward reducing low-confidence tokens only if text was not lost.
    if low_conf_delta < 0 and word_delta >= 0 and character_delta >= 0:
        score += abs(low_conf_delta) * 1.0

    return round(score, 3)



def edge_aware_sharpen(image, strength=1.0, blur_sigma=1.0, edge_percentile=70):
    if strength <= 0:
        return image.copy()

    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    detail = img - blurred

    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    threshold = np.percentile(magnitude, edge_percentile)
    edge_mask = (magnitude > threshold).astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 1.0)
    edge_mask = np.clip(edge_mask, 0.0, 1.0)

    sharpened = img + (detail * strength * edge_mask)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def post_process_document(
    denoised_image,
    original_image=None,
    profile="human",
    apply_thresholding=False,
    blend_factor=0.15,
    morph_kernel_size=0,
    sharpen_level=1,
):
    """
    human profile:
        softer, more natural looking

    ocr profile:
        stronger local contrast and more decisive text edges
    """
    processed = denoised_image.copy()

    if original_image is not None and blend_factor > 0:
        original_weight = float(np.clip(blend_factor, 0.0, 0.5))
        denoised_weight = 1.0 - original_weight
        processed = cv2.addWeighted(
            processed,
            denoised_weight,
            original_image,
            original_weight,
            0,
        )

    if profile == "ocr":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    if apply_thresholding:
        processed = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            7,
        )

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    if sharpen_level > 0:
        if sharpen_level == 1:
            sharp_kernel = np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                dtype=np.float32,
            )
        elif sharpen_level == 2:
            sharp_kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                dtype=np.float32,
            )
        else:
            sharp_kernel = np.array(
                [[-2, -2, -2], [-2, 13, -2], [-2, -2, -2]],
                dtype=np.float32,
            )

        sharpened = cv2.filter2D(processed, -1, sharp_kernel)

        edges = cv2.Canny(processed, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edge_mask = edges.astype(np.float32) / 255.0

        edge_strength = 0.65 if profile == "human" else 0.9
        edge_mask = np.clip(edge_mask * edge_strength, 0.0, 1.0)

        processed_f = processed.astype(np.float32)
        sharpened_f = sharpened.astype(np.float32)

        processed = (processed_f * (1.0 - edge_mask)) + (
            sharpened_f * edge_mask
        )
        processed = np.clip(processed, 0, 255).astype(np.uint8)

    if profile == "ocr" and not apply_thresholding:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)

    return np.clip(processed, 0, 255).astype(np.uint8)


def generate_dual_outputs(denoised_image, original_image=None, auto_tune=True):
    if auto_tune:
        human_image = auto_tune_parameters(
            denoised_image,
            original_image=original_image,
            profile="human",
        )
        ocr_image = auto_tune_parameters(
            denoised_image,
            original_image=original_image,
            profile="ocr",
        )
    else:
        human_image = post_process_document(
            denoised_image,
            original_image=original_image,
            profile="human",
            apply_thresholding=False,
            blend_factor=0.12,
            morph_kernel_size=0,
            sharpen_level=2,
        )
        ocr_image = post_process_document(
            denoised_image,
            original_image=original_image,
            profile="ocr",
            apply_thresholding=True,
            blend_factor=0.08,
            morph_kernel_size=0,
            sharpen_level=2,
        )

    return {
        "human": human_image,
        "ocr": ocr_image,
    }


def save_as_pdf(image_path, pdf_path):
    image = Image.open(image_path).convert("L")
    image.save(pdf_path, "PDF", resolution=300.0)
    print(f"📄 Saved PDF: {pdf_path}")


def auto_select_best_weight(weights_folder, sample_image):
    configure_tesseract()
    mat_files = validate_weights_folder(weights_folder)

    original_image = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(
            f"Sample image not found or unreadable: {sample_image}"
        )

    best_quality = -1.0
    best_weight = None

    for weight_file in mat_files:
        model = DnCNN(channels=1, num_of_layers=17)
        weight_path = os.path.join(weights_folder, weight_file)

        load_h5_weights(weight_path, model)
        model.eval()

        denoised_image = denoise_with_cnn(model, sample_image)

        processed_image = post_process_document(
            denoised_image,
            original_image=original_image,
            profile="ocr",
            apply_thresholding=False,
            blend_factor=0.12,
            morph_kernel_size=0,
            sharpen_level=1,
        )

        text_quality = ocr_text_quality(processed_image, profile="ocr")
        print(f"🔍 Tested {weight_file}: OCR score={text_quality:.2f}")

        if text_quality > best_quality:
            best_quality = text_quality
            best_weight = weight_file

    if best_weight is None:
        raise RuntimeError("Could not select a best weight file.")

    print(f"✅ Best weight selected: {best_weight} OCR score={best_quality:.2f}")
    return best_weight


def auto_tune_parameters(denoised_image, original_image=None, profile="human"):
    configure_tesseract()

    best_score = -1.0
    best_params = None
    best_image = None

    if profile == "human":
        blend_factors = [0.10, 0.15, 0.20]
        sharpen_levels = [0, 1, 2]
        thresholding_options = [False]
        morph_kernel_sizes = [0]
    else:
        blend_factors = [0.05, 0.10, 0.15]
        sharpen_levels = [1, 2]
        thresholding_options = [False]
        morph_kernel_sizes = [0]

    for blend in blend_factors:
        for sharpen in sharpen_levels:
            for threshold in thresholding_options:
                for morph_kernel in morph_kernel_sizes:
                    processed_image = post_process_document(
                        denoised_image,
                        original_image=original_image,
                        profile=profile,
                        apply_thresholding=threshold,
                        blend_factor=blend,
                        morph_kernel_size=morph_kernel,
                        sharpen_level=sharpen,
                    )

                    score = ocr_text_quality(processed_image, profile=profile)

                    if score > best_score:
                        best_score = score
                        best_params = (blend, sharpen, threshold, morph_kernel)
                        best_image = processed_image

    if best_params is None or best_image is None:
        raise RuntimeError(f"Auto-tuning failed for profile={profile}")

    print(
        f"✅ Auto-selected parameters for {profile}: "
        f"Blend={best_params[0]}, "
        f"Sharpen={best_params[1]}, "
        f"Threshold={best_params[2]}, "
        f"Morphology={best_params[3]}, "
        f"Score={best_score:.2f}"
    )

    return best_image


def batch_clean_documents(
    weights_path,
    input_folder,
    output_folder,
    auto_tune=True,
    make_dual_output=True,
    engine="cnn",
    sbb_model_dir="external_models/sbb_binarization/saved_model",
    sbb_conda_env="sbb310",
):
    configure_tesseract()
    validate_input_folder(input_folder)

    if engine not in {"cnn", "sbb", "auto"}:
        raise ValueError(
            f"Unsupported engine: {engine}. Expected 'cnn', 'sbb', or 'auto'."
        )

    model = None
    if engine in {"cnn", "auto"}:
        model = DnCNN(channels=1, num_of_layers=17)
        load_h5_weights(weights_path, model)
        model.eval()

    os.makedirs(output_folder, exist_ok=True)

    job_id = str(uuid.uuid4())
    processed_files = []
    failed_files = []
    manifest_outputs = {}
    manifest_metrics = {}
    manifest_errors = []

    for filename in sorted(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            base_name = os.path.splitext(filename)[0]
            print(f"Processing: {filename}")

            original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise FileNotFoundError(
                    f"Input image not found or unreadable: {file_path}"
                )

            metrics_before = ocr_metrics(original_image)

            candidate_metrics = {}
            selected_engine_for_file = engine

            if engine == "cnn":
                denoised_image = denoise_with_cnn(model, file_path)

                if make_dual_output:
                    outputs = generate_dual_outputs(
                        denoised_image,
                        original_image=original_image,
                        auto_tune=auto_tune,
                    )
                else:
                    outputs = {
                        "human": post_process_document(
                            denoised_image,
                            original_image=original_image,
                            profile="human",
                            apply_thresholding=False,
                            blend_factor=0.12,
                            morph_kernel_size=0,
                            sharpen_level=2,
                        )
                    }

            elif engine == "sbb":
                sbb_image_path = os.path.join(
                    output_folder,
                    f"{base_name}_sbb.png",
                )
                run_sbb_binarization(
                    input_path=file_path,
                    output_path=sbb_image_path,
                    model_dir=sbb_model_dir,
                    conda_env=sbb_conda_env,
                )

                sbb_image = cv2.imread(sbb_image_path, cv2.IMREAD_GRAYSCALE)
                if sbb_image is None:
                    raise RuntimeError(f"SBB output was not readable: {sbb_image_path}")

                outputs = {
                    "human": sbb_image,
                    "ocr": sbb_image,
                }

            else:
                candidates = {}

                denoised_image = denoise_with_cnn(model, file_path)
                cnn_outputs = generate_dual_outputs(
                    denoised_image,
                    original_image=original_image,
                    auto_tune=auto_tune,
                )

                cnn_ocr_output = (
                    cnn_outputs["ocr"]
                    if "ocr" in cnn_outputs
                    else cnn_outputs["human"]
                )
                cnn_after_metrics = ocr_metrics(cnn_ocr_output)
                cnn_comparison = compare_ocr_metrics(
                    metrics_before,
                    cnn_after_metrics,
                )
                cnn_score = score_ocr_candidate(cnn_comparison)

                candidates["cnn"] = {
                    "outputs": cnn_outputs,
                    "after": cnn_after_metrics,
                    "comparison": cnn_comparison,
                    "score": cnn_score,
                }

                sbb_image_path = os.path.join(
                    output_folder,
                    f"{base_name}_sbb_candidate.png",
                )
                run_sbb_binarization(
                    input_path=file_path,
                    output_path=sbb_image_path,
                    model_dir=sbb_model_dir,
                    conda_env=sbb_conda_env,
                )

                sbb_image = cv2.imread(sbb_image_path, cv2.IMREAD_GRAYSCALE)
                if sbb_image is None:
                    raise RuntimeError(f"SBB output was not readable: {sbb_image_path}")

                sbb_outputs = {
                    "human": sbb_image,
                    "ocr": sbb_image,
                }
                sbb_after_metrics = ocr_metrics(sbb_image)
                sbb_comparison = compare_ocr_metrics(
                    metrics_before,
                    sbb_after_metrics,
                )
                sbb_score = score_ocr_candidate(sbb_comparison)

                candidates["sbb"] = {
                    "outputs": sbb_outputs,
                    "after": sbb_after_metrics,
                    "comparison": sbb_comparison,
                    "score": sbb_score,
                }

                selected_engine_for_file = max(
                    candidates,
                    key=lambda name: candidates[name]["score"],
                )
                outputs = candidates[selected_engine_for_file]["outputs"]

                candidate_metrics = {
                    name: {
                        "score": data["score"],
                        "after": data["after"],
                        **data["comparison"],
                    }
                    for name, data in candidates.items()
                }

                print(
                    f"🤖 Auto selected {selected_engine_for_file} for {filename} "
                    f"(cnn={cnn_score}, sbb={sbb_score})"
                )

            manifest_outputs[filename] = {}

            for variant_name, final_image in outputs.items():
                cleaned_image_path = os.path.join(
                    output_folder,
                    f"{base_name}_cleaned_{variant_name}.png",
                )
                pdf_output_path = os.path.join(
                    output_folder,
                    f"{base_name}_{variant_name}.pdf",
                )

                cv2.imwrite(cleaned_image_path, final_image)
                save_as_pdf(cleaned_image_path, pdf_output_path)

                manifest_outputs[filename][variant_name] = {
                    "image": cleaned_image_path,
                    "pdf": pdf_output_path,
                }

            ocr_output = outputs["ocr"] if "ocr" in outputs else outputs["human"]
            metrics_after = ocr_metrics(ocr_output)

            delta_mean_confidence = round(
                metrics_after["mean_confidence"] - metrics_before["mean_confidence"],
                3,
            )
            delta_extracted_words = (
                metrics_after["extracted_words"] - metrics_before["extracted_words"]
            )
            delta_extracted_characters = (
                metrics_after["extracted_characters"]
                - metrics_before["extracted_characters"]
            )
            delta_low_confidence_tokens = (
                metrics_after["low_confidence_tokens"]
                - metrics_before["low_confidence_tokens"]
            )

            ocr_improved = (
                delta_mean_confidence > 0
                and delta_extracted_words >= 0
            )

            if ocr_improved and delta_low_confidence_tokens <= 0:
                ocr_quality_note = (
                    "OCR confidence improved and low-confidence token count did not increase."
                )
            elif ocr_improved:
                ocr_quality_note = (
                    "OCR confidence and extracted word count improved, "
                    "but low-confidence token count also increased."
                )
            else:
                ocr_quality_note = (
                    "OCR metrics did not clearly improve. "
                    "Review output before using as OCR-optimized result."
                )

            comparison = compare_ocr_metrics(metrics_before, metrics_after)

            manifest_metrics[filename] = {
                "before": metrics_before,
                "after": metrics_after,
                **comparison,
                "selected_engine": selected_engine_for_file,
            }

            if candidate_metrics:
                manifest_metrics[filename]["candidate_metrics"] = candidate_metrics

            processed_files.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append((filename, str(e)))
            manifest_errors.append(
                {
                    "file": filename,
                    "error": str(e),
                }
            )

    if engine == "cnn":
        selected_profile = (
            "cnn+dual-output+auto-tune"
            if auto_tune and make_dual_output
            else "cnn+single-output"
            if not make_dual_output
            else "cnn+dual-output"
        )
        model_info = {
            "model_type": "DnCNN",
            "weights_path": weights_path,
        }
        steps = [
            "load_dncnn_model",
            "read_grayscale_image",
            "measure_ocr_before",
            "cnn_denoise",
            "generate_human_output",
            "generate_ocr_output" if make_dual_output else "generate_single_output",
            "save_png",
            "save_pdf",
            "measure_ocr_after",
        ]
    elif engine == "sbb":
        selected_profile = "sbb-binarization"
        model_info = {
            "model_type": "SBB Binarization",
            "model_dir": sbb_model_dir,
            "conda_env": sbb_conda_env,
        }
        steps = [
            "read_grayscale_image",
            "measure_ocr_before",
            "sbb_binarize",
            "save_png",
            "save_pdf",
            "measure_ocr_after",
        ]
    else:
        selected_profile = "auto-cnn-vs-sbb"
        model_info = {
            "candidate_engines": {
                "cnn": {
                    "model_type": "DnCNN",
                    "weights_path": weights_path,
                },
                "sbb": {
                    "model_type": "SBB Binarization",
                    "model_dir": sbb_model_dir,
                    "conda_env": sbb_conda_env,
                },
            }
        }
        steps = [
            "load_dncnn_model",
            "read_grayscale_image",
            "measure_ocr_before",
            "cnn_candidate",
            "sbb_candidate",
            "score_candidates",
            "select_best_engine_per_file",
            "save_png",
            "save_pdf",
            "measure_ocr_after",
        ]

    manifest = build_manifest(
        job_id=job_id,
        engine=engine,
        selected_profile=selected_profile,
        input_file=input_folder,
        outputs=manifest_outputs,
        metrics=manifest_metrics,
        model=model_info,
        steps=steps,
        errors=manifest_errors,
    )

    manifest_path = write_manifest(manifest, output_folder)

    return {
        "processed": processed_files,
        "failed": failed_files,
        "manifest": manifest_path,
    }


def main():
    print("🚀 Running Document Cleaning CLI...")

    parser = argparse.ArgumentParser(
        description=(
            "Document Image Denoising CLI with OCR-aware auto-tuning "
            "and dual human/OCR outputs"
        )
    )

    parser.add_argument(
        "weights_folder",
        type=str,
        help="Folder containing weight files .mat",
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing noisy document images",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder to save cleaned documents",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Automatically tune post-processing parameters",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically select the best weight file",
    )
    parser.add_argument(
        "--single-output",
        action="store_true",
        help="Only save the human-readable output instead of both human and OCR outputs",
    )
    parser.add_argument(
        "--engine",
        choices=["cnn", "sbb", "auto"],
        default="cnn",
        help="Cleaning engine to use. Default: cnn",
    )
    parser.add_argument(
        "--sbb-model-dir",
        default="external_models/sbb_binarization/saved_model",
        help="SBB model directory. Used only with --engine sbb.",
    )
    parser.add_argument(
        "--sbb-conda-env",
        default="sbb310",
        help="Conda environment containing sbb_binarize. Used only with --engine sbb.",
    )

    args = parser.parse_args()

    try:
        configure_tesseract()
        mat_files = validate_weights_folder(args.weights_folder)
        validate_input_folder(args.input_folder)

        if args.engine in {"cnn", "auto"} and args.auto_select:
            print("🔍 Auto-selecting best CNN weight...")
            sample_image = next(
                os.path.join(args.input_folder, f)
                for f in sorted(os.listdir(args.input_folder))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            best_weight = auto_select_best_weight(args.weights_folder, sample_image)
        else:
            best_weight = "sigma=10.mat" if "sigma=10.mat" in mat_files else mat_files[0]
            if args.engine in {"cnn", "auto"} and best_weight != "sigma=10.mat":
                print(
                    f"⚠️ Default weight sigma=10.mat not found. "
                    f"Falling back to: {best_weight}"
                )

        weights_path = os.path.join(args.weights_folder, best_weight)

        print(f"🧰 Engine: {args.engine}")
        if args.engine in {"cnn", "auto"}:
            print(f"📂 Using weights: {weights_path}")
        if args.engine in {"sbb", "auto"}:
            print(f"📂 Using SBB model dir: {args.sbb_model_dir}")
            print(f"🐍 Using SBB conda env: {args.sbb_conda_env}")
        print(f"📥 Input folder: {args.input_folder}")
        print(f"📤 Output folder: {args.output_folder}")

        result = batch_clean_documents(
            weights_path=weights_path,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            auto_tune=args.auto_tune,
            make_dual_output=not args.single_output,
            engine=args.engine,
            sbb_model_dir=args.sbb_model_dir,
            sbb_conda_env=args.sbb_conda_env,
        )

        print(
            f"✅ Document Cleaning Complete! "
            f"Processed: {len(result['processed'])}, "
            f"Failed: {len(result['failed'])}"
        )
        print(f"🧾 Manifest saved to: {result['manifest']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

