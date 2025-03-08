import torch
import numpy as np
import cv2
from models import DnCNN  # DnCNN model from the repository

# Load pre-trained model
model = DnCNN(channels=1, num_of_layers=17)  # Define the model
model.load_state_dict(torch.load('./models/DnCNN_sigma25.pth'))  # Load weights
model.eval()

def denoise_image(model, noisy_image_path, output_path):
    # Load noisy image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    noisy_tensor = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Denoise
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor).squeeze().numpy()

    # Convert back to image
    denoised_image = (denoised_tensor * 255).astype(np.uint8)
    cv2.imwrite(output_path, denoised_image)
    print(f"Saved denoised image to {output_path}")

# Example usage
noisy_image_path = "data/val/noisy_0.png"  # Path to noisy image
output_image_path = "denoised_image.png"  # Output path
denoise_image(model, noisy_image_path, output_image_path)
