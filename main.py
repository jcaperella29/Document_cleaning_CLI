import os
import cv2
import torch
import base64
import shutil
import uvicorn
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from processor import batch_clean_documents, auto_select_best_weight

# FastAPI instance
app = FastAPI(title="Document Cleaner API")

# Folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed"
WEIGHTS_FOLDER = "model_weights"  # Updated path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Document Cleaning API is running!"}

@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    """Uploads and processes a document image."""
    # Paths
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_img_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}_cleaned.png")
    output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}.pdf")

    # Save uploaded image
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔍 Auto-select best weight using OCR-based logic
    best_weight_file = auto_select_best_weight(WEIGHTS_FOLDER, input_path)
    weights_path = os.path.join(WEIGHTS_FOLDER, best_weight_file)

    print(f"✅ Using best weight: {weights_path}")

    # Run document cleaning pipeline
    batch_clean_documents(
        weights_path=weights_path,
        input_folder=UPLOAD_FOLDER,
        output_folder=OUTPUT_FOLDER,
        auto_tune=True
    )

    # Encode cleaned image to Base64
    with open(output_img_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return {
        "message": "Processing complete",
        "best_weight": best_weight_file,
        "cleaned_image_base64": encoded_img,
        "download_pdf": f"/download/{Path(file.filename).stem}.pdf"
    }

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """Serves the processed PDF file for download."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
