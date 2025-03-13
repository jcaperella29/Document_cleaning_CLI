import os
import argparse
import cv2
import h5py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pytesseract
from skimage.metrics import peak_signal_noise_ratio as psnr

import sys

# ✅ Auto-detect platform and set tesseract path safely
if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# DnCNN Model
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

# Load weights
def load_h5_weights(mat_file_path, model):
    with h5py.File(mat_file_path, 'r') as f:
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

# Denoising
def denoise_with_cnn(model, noisy_image_path):
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    if noisy_image is None:
        raise FileNotFoundError(f"Noisy image not found: {noisy_image_path}")
    noisy_image = noisy_image / 255.0
    noisy_tensor = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze(0).squeeze(0).numpy()
    return (denoised_tensor * 255).astype(np.uint8)

# OCR Quality Check
def ocr_text_quality(image):
    """Extract text using Tesseract OCR and measure quality."""
    text = pytesseract.image_to_string(image, config="--psm 6")
    return len(text.strip())  # More text = better quality
def auto_select_best_weight(weights_folder, sample_image):
    best_quality = -1
    best_weight = None
    model = DnCNN(channels=1, num_of_layers=17)

    original_image = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)

    for weight_file in os.listdir(weights_folder):
        if weight_file.endswith(".mat"):
            weight_path = os.path.join(weights_folder, weight_file)
            load_h5_weights(weight_path, model)
            model.eval()

            # Apply denoising
            denoised_image = denoise_with_cnn(model, sample_image)

            # ✅ Pass all required arguments
            processed_image = post_process_document(
                denoised_image, 
                apply_thresholding=False, 
                blend_factor=0.3, 
                morph_kernel_size=0, 
                sharpen_level=1
            )

            # Evaluate OCR text extraction
            text_quality = ocr_text_quality(processed_image)

            print(f"🔍 Tested {weight_file}: Extracted Text Length={text_quality}")

            if text_quality > best_quality:
                best_quality = text_quality
                best_weight = weight_file

    print(f"✅ Best weight selected: {best_weight} (OCR Score={best_quality})")
    return best_weight


def post_process_document(denoised_image, apply_thresholding, blend_factor, morph_kernel_size, sharpen_level):
    if apply_thresholding:
        denoised_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 25, 2)
    
    if blend_factor > 0:
        denoised_image = cv2.addWeighted(denoised_image, blend_factor, denoised_image, 1 - blend_factor, 2)

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

    # ✅ Auto-tune sharpening level
    if sharpen_level == 1:  # Light sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    elif sharpen_level == 2:  # Medium sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    elif sharpen_level == 3:  # Strong sharpening
        kernel = np.array([[-2, -2, -2], [-2, 13, -2], [-2, -2, -2]], dtype=np.float32)
    else:
        return denoised_image  # No sharpening

    denoised_image = cv2.filter2D(denoised_image, -1, kernel)
    
    return denoised_image


# Save Processed Image as PDF
def save_as_pdf(image_path, pdf_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image.save(pdf_path, "PDF", resolution=300.0)
    print(f"📄 Saved PDF: {pdf_path}")

def auto_tune_parameters(denoised_image):
    best_score = -1
    best_params = None
    best_image = None

    # ✅ Define parameter search space
    blend_factors = [0.2, 0.3, 0.4]
    sharpen_levels = [0, 1, 2]  # 0 = No sharpening, 1 = Light, 2 = Medium
    thresholding_options = [False, True]
    morph_kernel_sizes = [0]  # Keep morphology disabled for now

    for blend in blend_factors:
        for sharpen in sharpen_levels:
            for threshold in thresholding_options:
                for morph_kernel in morph_kernel_sizes:
                    # ✅ Pass all required arguments correctly
                    processed_image = post_process_document(
                        denoised_image, 
                        apply_thresholding=threshold, 
                        blend_factor=blend, 
                        morph_kernel_size=morph_kernel, 
                        sharpen_level=sharpen
                    )

                    # ✅ Evaluate OCR text quality
                    text = pytesseract.image_to_string(processed_image, config="--psm 6")
                    score = len(text.strip())  # More extracted text = better settings

                    if score > best_score:
                        best_score = score
                        best_params = (blend, sharpen, threshold, morph_kernel)
                        best_image = processed_image

    print(f"✅ Auto-selected parameters: Blend={best_params[0]}, Sharpen={best_params[1]}, Threshold={best_params[2]}, Morphology={best_params[3]}")
    return best_image

def batch_clean_documents(weights_path, input_folder, output_folder, auto_tune=False):
    model = DnCNN(channels=1, num_of_layers=17)
    load_h5_weights(weights_path, model)
    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            base_name = os.path.splitext(filename)[0]
            cleaned_image_path = os.path.join(output_folder, f"{base_name}_cleaned.png")
            pdf_output_path = os.path.join(output_folder, f"{base_name}.pdf")

            print(f"Processing: {filename}")
            denoised_image = denoise_with_cnn(model, file_path)

            # ✅ Auto-tune parameters for best OCR results
            if auto_tune:
                final_image = auto_tune_parameters(denoised_image)
            else:
                # ✅ Provide default parameters if auto-tune is OFF
                final_image = post_process_document(
                    denoised_image, 
                    apply_thresholding=False, 
                    blend_factor=0.3, 
                    morph_kernel_size=0, 
                    sharpen_level=1
                )

            cv2.imwrite(cleaned_image_path, final_image)
            save_as_pdf(cleaned_image_path, pdf_output_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def main():
    print("🚀 Running Document Cleaning CLI...")

    parser = argparse.ArgumentParser(description="Document Image Denoising CLI with Auto-Weight Selection")
    parser.add_argument("weights_folder", type=str, help="Folder containing weight files (.mat)")
    parser.add_argument("input_folder", type=str, help="Folder containing noisy document images")
    parser.add_argument("output_folder", type=str, help="Folder to save cleaned documents")
    parser.add_argument("--auto-tune", action="store_true", help="Automatically tune post-processing parameters")

    # ✅ Add missing flags
    parser.add_argument("--auto-select", action="store_true", help="Automatically select the best weight file")
    parser.add_argument("--disable-thresholding", action="store_true", help="Disable adaptive thresholding")
    parser.add_argument("--blend-factor", type=float, default=0.8, help="Blending strength (0.0 - 1.0)")
    parser.add_argument("--morph-kernel", type=int, default=2, help="Morphological kernel size (0 to disable)")
    parser.add_argument("--sharpen", action="store_true", help="Apply image sharpening to enhance text clarity")

    args = parser.parse_args()

    # Auto-select weight file if enabled
    if args.auto_select:
        print("🔍 Auto-selecting best weight...")
        sample_image = next((os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))), None)
        if not sample_image:
            print("❌ No valid sample image found in input folder.")
            return
        best_weight = auto_select_best_weight(args.weights_folder, sample_image)
    else:
        best_weight = "sigma=10.mat"  # Default if not auto-selecting

    weights_path = os.path.join(args.weights_folder, best_weight)

    print(f"📂 Using weights: {weights_path}")
    print(f"📥 Input folder: {args.input_folder}")
    print(f"📤 Output folder: {args.output_folder}")

    batch_clean_documents(
    weights_path, 
    args.input_folder, 
    args.output_folder, 
    auto_tune=args.auto_tune  # ✅ Pass auto-tune flag properly
)


    print("✅ Document Cleaning Complete!")

if __name__ == "__main__":
    main()
