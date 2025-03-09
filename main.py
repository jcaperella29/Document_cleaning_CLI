import os
import cv2
import torch
import base64
import shutil
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from processor import batch_clean_documents  # Import your CLI logic
from pathlib import Path

app = FastAPI(title="Document Cleaner API")

# Create necessary directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Document Cleaning API is running!"}


@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    """Uploads and processes a document image."""
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_img_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}_cleaned.png")
    output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}.pdf")

    # Save uploaded image
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run document cleaning pipeline
    batch_clean_documents(
        weights_path="weights/sigma=10.mat",
        input_folder=UPLOAD_FOLDER,
        output_folder=OUTPUT_FOLDER,
        auto_tune=True
    )

    # Convert image to base64 for API response
    with open(output_img_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    return {
        "message": "Processing complete",
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

