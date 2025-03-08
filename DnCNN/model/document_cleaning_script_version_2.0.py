import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image


# ---------------------
# Define DnCNN Model
# ---------------------
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # Initial convolutional layer
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Intermediate convolutional layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # Final convolutional layer
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning


# ---------------------
# Denoising Function
# ---------------------
def denoise_with_cnn(model, noisy_image_path):
    # Load the noisy image as a grayscale image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    if noisy_image is None:
        raise FileNotFoundError(f"Noisy image not found at: {noisy_image_path}")

    # Normalize the image to the range [0, 1] and convert to a tensor
    noisy_image = noisy_image / 255.0
    noisy_tensor = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform denoising using the DNCNN model
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions

    # Convert the denoised tensor back to an image
    denoised_image = (denoised_tensor * 255).astype(np.uint8)
    return denoised_image


# ---------------------
# Post-Processing Steps
# ---------------------
def post_process_document(denoised_image):
    # Step 1: Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

    # Step 2: Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    return cleaned


# ---------------------
# PDF Conversion
# ---------------------
def convert_to_pdf(image_path, pdf_path):
    # Open the cleaned image using Pillow
    image = Image.open(image_path)

    # Ensure it's in RGB mode (required for PDF conversion)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save the image as a PDF
    image.save(pdf_path, "PDF", resolution=100.0)
    print(f"Cleaned document saved as PDF at {pdf_path}")


# ---------------------
# Main Pipeline
# ---------------------
def clean_document_pipeline(model_path, noisy_image_path, cleaned_image_path, pdf_output_path):
    # Step 1: Load the pre-trained DNCNN model
    model = torch.load(model_path, map_location=torch.device("cpu"))  # Load pre-trained model
    model.eval()  # Set the model to evaluation mode

    # Step 2: Denoise the image using the CNN
    denoised_image = denoise_with_cnn(model, noisy_image_path)

    # Step 3: Post-process the denoised image
    final_cleaned_image = post_process_document(denoised_image)

    # Step 4: Save the cleaned image
    cv2.imwrite(cleaned_image_path, final_cleaned_image)
    print(f"Final cleaned document saved to {cleaned_image_path}")

    # Step 5: Convert the cleaned image to PDF
    convert_to_pdf(cleaned_image_path, pdf_output_path)


# ---------------------
# Run the Script
# ---------------------
if __name__ == "__main__":
    # Paths
    model_path = r"C:\Users\ccape\Downloads\document cleaning\noise2noise-pytorch\DnCNN\TrainingCodes\dncnn_pytorch\models\DnCNN_sigma25\model.pth"
    noisy_image_path = r"C:\Users\ccape\Downloads\document cleaning\noise2noise-pytorch\data\val\noisy_0.png"
    cleaned_image_path = r"C:\Users\ccape\Downloads/final_document.png"
    pdf_output_path = r"C:\Users\ccape\Downloads\final_document.pdf"

    # Execute the pipeline
    clean_document_pipeline(model_path, noisy_image_path, cleaned_image_path, pdf_output_path)
