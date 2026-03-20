import os
import shutil
import argparse

import cv2
import h5py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pytesseract


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
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out


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
        f for f in os.listdir(input_folder)
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
        raise FileNotFoundError(f"Noisy image not found or unreadable: {noisy_image_path}")

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
        prefers readable, crisp text without going too harsh
    ocr:
        prefers OCR extraction quality much more aggressively
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

    # Binary-ness score: OCR profile likes stronger black/white separation
    std_score = float(np.std(image)) / 255.0

    if profile == "human":
        return (text_len * 1.0) + (mean_conf * 0.8) + (edge_score * 2.0)
    else:
        return (text_len * 1.2) + (mean_conf * 1.8) + (edge_score * 1.2) + (std_score * 8.0)
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

    # Restore some original detail for human readability
    if original_image is not None and blend_factor > 0:
        original_weight = float(np.clip(blend_factor, 0.0, 0.5))
        denoised_weight = 1.0 - original_weight
        processed = cv2.addWeighted(processed, denoised_weight, original_image, original_weight, 0)

    # OCR profile: boost local contrast before threshold/sharpen
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

    # Edge-aware sharpening
    if sharpen_level > 0:
        if sharpen_level == 1:
            sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        elif sharpen_level == 2:
            sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        else:
            sharp_kernel = np.array([[-2, -2, -2], [-2, 13, -2], [-2, -2, -2]], dtype=np.float32)

        sharpened = cv2.filter2D(processed, -1, sharp_kernel)

        # Build edge mask so flat background does not get over-sharpened
        edges = cv2.Canny(processed, 50, 150)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edge_mask = edges.astype(np.float32) / 255.0

        # OCR profile sharpens harder on edges
        edge_strength = 0.65 if profile == "human" else 0.9
        edge_mask = np.clip(edge_mask * edge_strength, 0.0, 1.0)

        processed_f = processed.astype(np.float32)
        sharpened_f = sharpened.astype(np.float32)

        processed = (processed_f * (1.0 - edge_mask) + sharpened_f * edge_mask)
        processed = np.clip(processed, 0, 255).astype(np.uint8)

    # OCR profile: optional final slight hardening of text/background separation
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
        raise FileNotFoundError(f"Sample image not found or unreadable: {sample_image}")

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
            apply_thresholding=False,
            blend_factor=0.12,
            morph_kernel_size=0,
            sharpen_level=1,
            edge_aware=True,
        )

        text_quality = ocr_text_quality(processed_image)
        print(f"🔍 Tested {weight_file}: OCR score={text_quality:.2f}")

        if text_quality > best_quality:
            best_quality = text_quality
            best_weight = weight_file

    if best_weight is None:
        raise RuntimeError("Could not select a best weight file.")

    print(f"✅ Best weight selected: {best_weight} (OCR score={best_quality:.2f})")
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
        blend_factors = [0.0, 0.05, 0.10]
        sharpen_levels = [1, 2, 3]
        thresholding_options = [False, True]
        morph_kernel_sizes = [0, 1]

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
        f"Blend={best_params[0]}, Sharpen={best_params[1]}, "
        f"Threshold={best_params[2]}, Morphology={best_params[3]}, "
        f"Score={best_score:.2f}"
    )
    return best_image

def batch_clean_documents(weights_path, input_folder, output_folder, auto_tune=True, make_dual_output=True):
    configure_tesseract()
    validate_input_folder(input_folder)

    model = DnCNN(channels=1, num_of_layers=17)
    load_h5_weights(weights_path, model)
    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    processed_files = []
    failed_files = []

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
                raise FileNotFoundError(f"Input image not found or unreadable: {file_path}")

            denoised_image = denoise_with_cnn(model, file_path)

            outputs = generate_dual_outputs(
                denoised_image,
                original_image=original_image,
                auto_tune=auto_tune,
            ) if make_dual_output else {
                "human": post_process_document(
                    denoised_image,
                    original_image=original_image,
                    apply_thresholding=False,
                    blend_factor=0.12,
                    morph_kernel_size=0,
                    sharpen_level=2,
                    edge_aware=True,
                )
            }

            for variant_name, final_image in outputs.items():
                cleaned_image_path = os.path.join(output_folder, f"{base_name}_cleaned_{variant_name}.png")
                pdf_output_path = os.path.join(output_folder, f"{base_name}_{variant_name}.pdf")
                cv2.imwrite(cleaned_image_path, final_image)
                save_as_pdf(cleaned_image_path, pdf_output_path)

            processed_files.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append((filename, str(e)))

    return {
        "processed": processed_files,
        "failed": failed_files,
    }


def main():
    print("🚀 Running Document Cleaning CLI...")

    parser = argparse.ArgumentParser(
        description="Document Image Denoising CLI with edge-aware sharpening and dual outputs"
    )
    parser.add_argument("weights_folder", type=str, help="Folder containing weight files (.mat)")
    parser.add_argument("input_folder", type=str, help="Folder containing noisy document images")
    parser.add_argument("output_folder", type=str, help="Folder to save cleaned documents")
    parser.add_argument("--auto-tune", action="store_true", help="Automatically tune post-processing parameters")
    parser.add_argument("--auto-select", action="store_true", help="Automatically select the best weight file")
    parser.add_argument(
        "--single-output",
        action="store_true",
        help="Only save the human-readable output instead of both human and OCR outputs",
    )

    args = parser.parse_args()

    try:
        configure_tesseract()
        mat_files = validate_weights_folder(args.weights_folder)
        validate_input_folder(args.input_folder)

        if args.auto_select:
            print("🔍 Auto-selecting best weight...")
            sample_image = next(
                os.path.join(args.input_folder, f)
                for f in sorted(os.listdir(args.input_folder))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            best_weight = auto_select_best_weight(args.weights_folder, sample_image)
        else:
            best_weight = "sigma=10.mat" if "sigma=10.mat" in mat_files else mat_files[0]
            if best_weight != "sigma=10.mat":
                print(f"⚠️ Default weight sigma=10.mat not found. Falling back to: {best_weight}")

        weights_path = os.path.join(args.weights_folder, best_weight)

        print(f"📂 Using weights: {weights_path}")
        print(f"📥 Input folder: {args.input_folder}")
        print(f"📤 Output folder: {args.output_folder}")

        result = batch_clean_documents(
            weights_path,
            args.input_folder,
            args.output_folder,
            auto_tune=args.auto_tune,
            make_dual_output=not args.single_output,
        )

        print(
            f"✅ Document Cleaning Complete! "
            f"Processed: {len(result['processed'])}, Failed: {len(result['failed'])}"
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
