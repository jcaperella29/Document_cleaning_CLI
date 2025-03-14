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
from sklearn.metrics.pairwise import cosine_similarity

from processor import batch_clean_documents, auto_select_best_weight

# 🔧 GCP-Compatible Temp Paths
BASE_DIR = "/tmp"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Init directories
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


def are_images_similar(folder, threshold=0.85):
    vectors = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("L").resize((128, 128))
            vec = np.array(img).flatten()
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)

    if len(vectors) < 2:
        return True

    sim_matrix = cosine_similarity(vectors)
    sims = [sim_matrix[i][j] for i in range(len(vectors)) for j in range(i) if i != j]
    avg_similarity = np.mean(sims)
    print(f"🔬 Average image similarity: {avg_similarity:.2f}")
    return avg_similarity >= threshold


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

        is_similar = are_images_similar(extracted_input_folder)

        if is_similar:
            print("✅ Images are similar - using one shared best weight.")
            sample_image = next(
                (os.path.join(extracted_input_folder, f)
                 for f in os.listdir(extracted_input_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                None
            )
            if not sample_image:
                raise HTTPException(status_code=400, detail="No valid image found.")

            best_weight_file = auto_select_best_weight("model_weights", sample_image)
            weights_path = os.path.join("model_weights", best_weight_file)

            batch_clean_documents(
                weights_path=weights_path,
                input_folder=extracted_input_folder,
                output_folder=output_folder,
                auto_tune=True
            )
            result_note = "Images were similar - shared weight used."
        else:
            print("⚠️ Images differ - tuning weights per image.")
            for img_file in os.listdir(extracted_input_folder):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(extracted_input_folder, img_file)
                weight_file = auto_select_best_weight("model_weights", img_path)
                batch_clean_documents(
                    weights_path=os.path.join("model_weights", weight_file),
                    input_folder=extracted_input_folder,
                    output_folder=output_folder,
                    auto_tune=True
                )
            result_note = "Images were diverse - tuned per image (slower)."

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
