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

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import zipfile

@app.post("/process-batch/")
async def process_batch(file: UploadFile = File(...)):
    try:
        # Save uploaded zip
        input_path = "temp/input.zip"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process images
        output_zip_path = "temp/cleaned_docs.zip"
        run_cleaning_pipeline(input_path, output_zip_path)  # ⬅️ Make sure this finishes successfully

        # ✅ Check zip actually exists before returning
        if not os.path.exists(output_zip_path):
            raise HTTPException(status_code=500, detail="Output ZIP file not created")

        # ✅ Optional: check file size > 0
        if os.path.getsize(output_zip_path) < 1024:
            raise HTTPException(status_code=500, detail="Output ZIP too small — likely failed")

        return FileResponse(output_zip_path, media_type="application/zip", filename="cleaned_docs.zip")
    
    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """Download cleaned PDF by filename."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

