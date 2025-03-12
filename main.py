import os
import cv2
import torch
import base64
import shutil
import uvicorn
import tempfile
import numpy as np
from pathlib import Path
from zipfile import ZipFile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from processor import batch_clean_documents, auto_select_best_weight

# FastAPI instance
app = FastAPI(title="Document Cleaner API")

# Folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed"
WEIGHTS_FOLDER = "model_weights"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Document Cleaning API is running!"}

@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    """Uploads and processes a single document image."""
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_img_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}_cleaned.png")
    output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}.pdf")

    # Save uploaded image
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔍 Auto-select best weight
    best_weight_file = auto_select_best_weight(WEIGHTS_FOLDER, input_path)
    weights_path = os.path.join(WEIGHTS_FOLDER, best_weight_file)
    print(f"✅ Using best weight: {weights_path}")

    # Run pipeline
    batch_clean_documents(
        weights_path=weights_path,
        input_folder=UPLOAD_FOLDER,
        output_folder=OUTPUT_FOLDER,
        auto_tune=True
    )

    # Encode to base64
    with open(output_img_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return {
        "message": "Processing complete",
        "best_weight": best_weight_file,
        "cleaned_image_base64": encoded_img,
        "download_pdf": f"/download/{Path(file.filename).stem}.pdf"
    }

@app.post("/process-batch/")
async def process_batch(file: UploadFile = File(...)):
    """Accepts a zip file of images, returns a zip of processed outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract images
        batch_input_dir = os.path.join(temp_dir, "unzipped")
        os.makedirs(batch_input_dir, exist_ok=True)
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(batch_input_dir)

        # Clean output
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Select best model using first image
        first_img = next((f for f in os.listdir(batch_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
        if not first_img:
            return {"error": "No image files found in zip."}
        first_img_path = os.path.join(batch_input_dir, first_img)
        best_weight = auto_select_best_weight(WEIGHTS_FOLDER, first_img_path)
        weights_path = os.path.join(WEIGHTS_FOLDER, best_weight)

        # Process all images in the zip
        batch_clean_documents(
            weights_path=weights_path,
            input_folder=batch_input_dir,
            output_folder=OUTPUT_FOLDER,
            auto_tune=True
        )

        # Zip processed results
        zip_output_path = os.path.join(temp_dir, "processed_output.zip")
        with ZipFile(zip_output_path, 'w') as zipf:
            for root, _, files in os.walk(OUTPUT_FOLDER):
                for file_name in files:
                    abs_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(abs_path, OUTPUT_FOLDER)
                    zipf.write(abs_path, arcname=rel_path)

        return FileResponse(zip_output_path, filename="cleaned_batch_output.zip", media_type="application/zip")

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """Download cleaned PDF by filename."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

