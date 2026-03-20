import os
import uuid
import base64
import shutil
from pathlib import Path
from zipfile import ZipFile

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from processor import (
    DnCNN,
    load_h5_weights,
    denoise_with_cnn,
    generate_dual_outputs,
    batch_clean_documents,
    auto_select_best_weight,
)

BASE_DIR = "/tmp"
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
MODEL_WEIGHTS_DIR = "model_weights"
DEFAULT_WEIGHT_FILE = "sigma=20.mat"

os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Document Cleaning API")

# Global model loaded once at startup
model = None


@app.on_event("startup")
async def startup_event():
    global model
    print("🚀 FastAPI app started!")

    weight_path = os.path.join(MODEL_WEIGHTS_DIR, DEFAULT_WEIGHT_FILE)
    if not os.path.exists(weight_path):
        raise RuntimeError(f"Default weight file not found: {weight_path}")

    model = DnCNN(channels=1, num_of_layers=17)
    load_h5_weights(weight_path, model)
    model.eval()

    print(f"✅ Loaded model weights: {weight_path}")


def create_job_dirs():
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return job_id, job_dir, input_dir, output_dir


@app.get("/")
def home():
    return {"message": "Document Cleaning API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    temp_input = None

    try:
        job_id, job_dir, input_dir, output_dir = create_job_dirs()
        safe_name = Path(file.filename or "input.png").name
        temp_input = os.path.join(input_dir, safe_name)

        contents = await file.read()
        with open(temp_input, "wb") as f:
            f.write(contents)

        original_image = cv2.imread(temp_input, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

        denoised_image = denoise_with_cnn(model, temp_input)

        outputs = generate_dual_outputs(
            denoised_image,
            original_image=original_image,
            auto_tune=True,
        )

        def encode_image(img):
            ok, buffer = cv2.imencode(".png", img)
            if not ok:
                raise ValueError("Failed to encode image as PNG.")
            return base64.b64encode(buffer).decode("utf-8")

        response = {
            "message": "Processing complete",
            "job_id": job_id,
            "best_weight": DEFAULT_WEIGHT_FILE,
            "outputs": {
                "human": {
                    "image_base64": encode_image(outputs["human"]),
                    "image_filename": f"{Path(safe_name).stem}_cleaned_human.png",
                    "pdf_filename": f"{Path(safe_name).stem}_human.pdf",
                },
                "ocr": {
                    "image_base64": encode_image(outputs["ocr"]),
                    "image_filename": f"{Path(safe_name).stem}_cleaned_ocr.png",
                    "pdf_filename": f"{Path(safe_name).stem}_ocr.pdf",
                },
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        if temp_input:
            shutil.rmtree(os.path.dirname(os.path.dirname(temp_input)), ignore_errors=True)


@app.post("/process-batch/")
async def process_batch(file: UploadFile = File(...)):
    job_id, job_dir, input_dir, output_dir = create_job_dirs()
    safe_name = Path(file.filename).name
    input_zip_path = os.path.join(job_dir, safe_name)
    output_zip_path = os.path.join(job_dir, "cleaned_docs.zip")

    try:
        with open(input_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with ZipFile(input_zip_path, "r") as zip_ref:
            zip_ref.extractall(input_dir)

        print("🧠 Sampling percentage of batch to auto-select best weight...")

        image_files = sorted(
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not image_files:
            raise HTTPException(status_code=400, detail="No valid images found in ZIP.")

        sample_percent = 0.2
        max_samples = 10
        num_samples = min(max(1, int(len(image_files) * sample_percent)), max_samples)
        sampled_images = image_files[:num_samples]

        print(f"📊 Sampling {num_samples} of {len(image_files)} total images...")

        weight_votes = []
        for img_file in sampled_images:
            img_path = os.path.join(input_dir, img_file)
            weight_file = auto_select_best_weight(MODEL_WEIGHTS_DIR, img_path)
            weight_votes.append(weight_file)
            print(f"🔍 {img_file} -> {weight_file}")

        best_weight_file = max(set(weight_votes), key=weight_votes.count)
        print(f"🏆 Selected shared weight: {best_weight_file}")

        batch_clean_documents(
            weights_path=os.path.join(MODEL_WEIGHTS_DIR, best_weight_file),
            input_folder=input_dir,
            output_folder=output_dir,
            auto_tune=True,
            make_dual_output=True,
        )

        result_note = (
            f"Sampled {num_samples}/{len(image_files)} images. "
            f"Using shared weight: {best_weight_file}. "
            f"Each image includes both human and OCR outputs by default."
        )

        with ZipFile(output_zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        if not os.path.exists(output_zip_path):
            raise HTTPException(status_code=500, detail="ZIP not created.")

        # Do NOT delete the job_dir here before the response is served.
        return FileResponse(
            output_zip_path,
            media_type="application/zip",
            filename="cleaned_docs.zip",
            headers={"X-Note": result_note},
        )

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
