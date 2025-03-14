import os
import cv2
import torch
import base64
import shutil
import tempfile
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from processor import batch_clean_documents, auto_select_best_weight

# 🔧 GCP-Compatible Temp Paths
BASE_DIR = "/tmp"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="Document Cleaning API")


@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI app started on GCP!")


@app.get("/")
def home():
    return {"message": "Document Cleaning API is running!"}


@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_img_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}_cleaned.png")
    output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{Path(file.filename).stem}.pdf")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    best_weight_file = auto_select_best_weight("model_weights", input_path)
    weights_path = os.path.join("model_weights", best_weight_file)
    print(f"✅ Using best weight: {best_weight_file}")

    batch_clean_documents(
        weights_path=weights_path,
        input_folder=UPLOAD_FOLDER,
        output_folder=OUTPUT_FOLDER,
        auto_tune=True
    )

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
    try:
        input_zip_path = os.path.join(TEMP_DIR, "input.zip")
        extracted_input_folder = os.path.join(TEMP_DIR, "unzipped_input")
        output_folder = os.path.join(TEMP_DIR, "processed_output")
        output_zip_path = os.path.join(TEMP_DIR, "cleaned_docs.zip")

        os.makedirs(extracted_input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        with open(input_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_input_folder)

        print("🧠 Sampling percentage of batch to auto-select best weight...")

        # Step 1: Gather valid image files
        image_files = [f for f in os.listdir(extracted_input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            raise HTTPException(status_code=400, detail="No valid images found in ZIP.")

        # Step 2: Adaptive sample logic
        sample_percent = 0.2
        max_samples = 10
        num_samples = min(max(1, int(len(image_files) * sample_percent)), max_samples)
        sampled_images = image_files[:num_samples]

        print(f"📊 Sampling {num_samples} of {len(image_files)} total images...")

        # Step 3: Select best weights
        weight_votes = []
        for img_file in sampled_images:
            img_path = os.path.join(extracted_input_folder, img_file)
            weight_file = auto_select_best_weight("model_weights", img_path)
            weight_votes.append(weight_file)
            print(f"🔍 {img_file} → {weight_file}")

        # Step 4: Pick most voted weight
        best_weight_file = max(set(weight_votes), key=weight_votes.count)
        print(f"🏆 Selected shared weight: {best_weight_file}")

        # Step 5: Run batch clean with shared weight
        batch_clean_documents(
            weights_path=os.path.join("model_weights", best_weight_file),
            input_folder=extracted_input_folder,
            output_folder=output_folder,
            auto_tune=True
        )

        result_note = f"Sampled {num_samples}/{len(image_files)} images. Using shared weight: {best_weight_file}"

        # Step 6: Zip up results
        with ZipFile(output_zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_folder):
                for f in files:
                    file_path = os.path.join(root, f)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)

        if not os.path.exists(output_zip_path):
            raise HTTPException(status_code=500, detail="ZIP not created")

        return FileResponse(
            output_zip_path,
            media_type="application/zip",
            filename="cleaned_docs.zip",
            headers={
                "X-Note": result_note.replace("—", "-")
            }
        )

    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# ☁️ Uvicorn boot removed for GCP Cloud Run
