import base64
import csv
import json
import logging
import mimetypes
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import FileResponse

from processor import (
    DnCNN,
    load_h5_weights,
    denoise_with_cnn,
    generate_dual_outputs,
    batch_clean_documents,
    auto_select_best_weight,
)

APP_VERSION = os.getenv("APP_VERSION", "0.2.4")
BASE_DIR = os.getenv("DOCUMENT_CLEANER_BASE_DIR", "/tmp")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
RESULT_ZIPS_DIR = os.path.join(BASE_DIR, "document_cleaner_result_zips")
MODEL_WEIGHTS_DIR = os.getenv("MODEL_WEIGHTS_DIR", "model_weights")
DEFAULT_WEIGHT_FILE = os.getenv("DEFAULT_WEIGHT_FILE", "sigma=20.mat")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "application/zip", "application/x-zip-compressed"}

MAX_SINGLE_UPLOAD_MB = int(os.getenv("MAX_SINGLE_UPLOAD_MB", "25"))
MAX_ZIP_UPLOAD_MB = int(os.getenv("MAX_ZIP_UPLOAD_MB", "250"))
MAX_FILES_PER_ZIP = int(os.getenv("MAX_FILES_PER_ZIP", "100"))
MAX_EXTRACTED_TOTAL_MB = int(os.getenv("MAX_EXTRACTED_TOTAL_MB", "500"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "12"))

MAX_SINGLE_UPLOAD_BYTES = MAX_SINGLE_UPLOAD_MB * 1024 * 1024
MAX_ZIP_UPLOAD_BYTES = MAX_ZIP_UPLOAD_MB * 1024 * 1024
MAX_EXTRACTED_TOTAL_BYTES = MAX_EXTRACTED_TOTAL_MB * 1024 * 1024

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULT_ZIPS_DIR, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("document_cleaner_api")

app = FastAPI(
    title="Document Cleaner API",
    version=APP_VERSION,
    description=(
        "Document-cleaning workflow API with human-readable and OCR-optimized "
        "outputs, safe ZIP handling, manifests, and batch QC artifacts."
    ),
)

model = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """
    Optional deployment gate.

    If DOCUMENT_CLEANER_API_KEY is set, every processing endpoint requires
    X-API-Key. If it is not set, the API remains open for local/dev use.
    """
    expected = os.getenv("DOCUMENT_CLEANER_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@app.on_event("startup")
async def startup_event() -> None:
    global model
    logger.info("FastAPI app starting", extra={"event": "startup"})
    cleanup_stale_jobs()

    weight_path = os.path.join(MODEL_WEIGHTS_DIR, DEFAULT_WEIGHT_FILE)
    if not os.path.exists(weight_path):
        raise RuntimeError(f"Default weight file not found: {weight_path}")

    model = DnCNN(channels=1, num_of_layers=17)
    load_h5_weights(weight_path, model)
    model.eval()
    logger.info("Loaded model weights", extra={"weight_path": weight_path})


def cleanup_stale_jobs() -> None:
    now = time.time()
    ttl_seconds = JOB_TTL_HOURS * 3600
    for child in Path(JOBS_DIR).iterdir():
        if not child.is_dir():
            continue
        try:
            if now - child.stat().st_mtime > ttl_seconds:
                shutil.rmtree(child, ignore_errors=True)
                logger.info("Deleted stale job directory", extra={"job_dir": str(child)})
        except Exception as exc:
            logger.warning("Failed stale job cleanup", extra={"job_dir": str(child), "error": str(exc)})


def cleanup_job_dir(job_dir: str) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)
    logger.info("Deleted job directory", extra={"job_dir": job_dir})


def cleanup_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted temporary response file", extra={"path": path})
    except Exception as exc:
        logger.warning("Failed temporary file cleanup", extra={"path": path, "error": str(exc)})


def create_job_dirs() -> Tuple[str, str, str, str]:
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return job_id, job_dir, input_dir, output_dir


def safe_basename(filename: Optional[str], fallback: str = "input.png") -> str:
    name = Path(filename or fallback).name
    if not name or name in {".", ".."}:
        name = fallback
    return name


def check_content_length(request: Request, max_bytes: int, label: str) -> None:
    header = request.headers.get("content-length")
    if not header:
        return
    try:
        size = int(header)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} upload is too large. Limit is {max_bytes // (1024 * 1024)} MB.",
        )


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds limit of {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_extension(filename: str, allowed: Iterable[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or '<none>'}")
    return ext


def validate_mime(upload: UploadFile, allowed: Iterable[str]) -> None:
    if upload.content_type and upload.content_type not in set(allowed):
        guessed = mimetypes.guess_type(upload.filename or "")[0]
        if guessed not in set(allowed):
            raise HTTPException(status_code=400, detail=f"Unsupported MIME type: {upload.content_type}")


def read_grayscale_image_checked(path: str) -> Any:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")
    height, width = image.shape[:2]
    pixels = int(height) * int(width)
    if pixels > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large: {pixels} pixels. Limit is {MAX_IMAGE_PIXELS}.",
        )
    return image


def encode_image_png(image: Any) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG.")
    return base64.b64encode(buffer).decode("utf-8")


def write_image_and_pdf(image: Any, png_path: str, pdf_path: str) -> None:
    from processor import save_as_pdf

    ok = cv2.imwrite(png_path, image)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {png_path}")
    save_as_pdf(png_path, pdf_path)


def build_run_manifest(
    *,
    job_id: str,
    selected_weight: str,
    files_received: int,
    files_processed: int,
    files_failed: int,
    processing_seconds: float,
    output_mode: str,
    failures: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": utc_now(),
        "api_version": APP_VERSION,
        "engine": "DnCNN",
        "selected_weight": selected_weight,
        "default_weight": DEFAULT_WEIGHT_FILE,
        "auto_tune": True,
        "dual_outputs": True,
        "files_received": files_received,
        "files_processed": files_processed,
        "files_failed": files_failed,
        "processing_seconds": round(processing_seconds, 3),
        "outputs": ["human_png", "ocr_png", "human_pdf", "ocr_pdf"],
        "output_mode": output_mode,
        "limits": {
            "max_single_upload_mb": MAX_SINGLE_UPLOAD_MB,
            "max_zip_mb": MAX_ZIP_UPLOAD_MB,
            "max_files": MAX_FILES_PER_ZIP,
            "max_extracted_total_mb": MAX_EXTRACTED_TOTAL_MB,
            "max_image_pixels": MAX_IMAGE_PIXELS,
        },
        "failures": failures or [],
    }


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def unique_flat_name(original_name: str, used: set[str]) -> str:
    base = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    clean_base = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in base).strip("_") or "image"
    candidate = f"{clean_base}{suffix}"
    i = 2
    while candidate in used:
        candidate = f"{clean_base}_{i}{suffix}"
        i += 1
    used.add(candidate)
    return candidate


def is_bad_zip_member_name(name: str) -> bool:
    pure = Path(name)
    parts = pure.parts
    return (
        name.startswith("/")
        or name.startswith("\\")
        or ".." in parts
        or any(part.startswith(".") for part in parts)
        or name.endswith("/")
    )


def safe_extract_zip_images(zip_path: str, dest_dir: str) -> List[Dict[str, Any]]:
    """
    Safely extracts supported images only, flattening nested paths.

    This rejects path traversal/absolute paths, hidden files/folders, too many images,
    oversized extracted totals, unsupported extensions, and unreadable/oversized images.
    """
    extracted: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    used_names: set[str] = set()
    total_bytes = 0

    try:
        with ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            image_members = [
                member for member in members
                if not member.is_dir() and Path(member.filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]

            if len(image_members) > MAX_FILES_PER_ZIP:
                raise HTTPException(
                    status_code=413,
                    detail=f"ZIP contains too many image files. Limit is {MAX_FILES_PER_ZIP}.",
                )

            for member in members:
                name = member.filename

                if member.is_dir():
                    continue

                if is_bad_zip_member_name(name):
                    failures.append({"file": name, "error": "Rejected unsafe ZIP path."})
                    continue

                ext = Path(name).suffix.lower()
                if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                    failures.append({"file": name, "error": "Skipped unsupported file type."})
                    continue

                total_bytes += int(member.file_size)
                if total_bytes > MAX_EXTRACTED_TOTAL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Extracted ZIP contents exceed {MAX_EXTRACTED_TOTAL_MB} MB.",
                    )

                flat_name = unique_flat_name(name, used_names)
                out_path = os.path.join(dest_dir, flat_name)

                with zip_ref.open(member, "r") as source, open(out_path, "wb") as target:
                    shutil.copyfileobj(source, target)

                try:
                    image = read_grayscale_image_checked(out_path)
                    extracted.append(
                        {
                            "original_name": name,
                            "input_filename": flat_name,
                            "path": out_path,
                            "width": int(image.shape[1]),
                            "height": int(image.shape[0]),
                            "pixels": int(image.shape[0]) * int(image.shape[1]),
                        }
                    )
                except HTTPException as exc:
                    os.remove(out_path)
                    failures.append({"file": name, "error": str(exc.detail)})

    except BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")

    if not extracted:
        detail = "No valid readable PNG/JPG/JPEG images found in ZIP."
        if failures:
            detail += f" First failure: {failures[0]['file']} - {failures[0]['error']}"
        raise HTTPException(status_code=400, detail=detail)

    failures_path = os.path.join(dest_dir, "_zip_validation_failures.json")
    write_json(failures_path, {"failures": failures})
    return extracted


def write_summary_files(output_dir: str, manifest_path: Optional[str], extra_failures: List[Dict[str, str]]) -> None:
    manifest_data: Dict[str, Any] = {}
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest_data = json.load(handle)
        except Exception as exc:
            logger.warning("Could not parse manifest for summary files", extra={"error": str(exc)})

    metrics = manifest_data.get("metrics") or manifest_data.get("per_file_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    errors = manifest_data.get("errors") or []
    if not isinstance(errors, list):
        errors = []
    errors.extend(extra_failures or [])

    summary_csv = os.path.join(output_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "selected_engine",
                "mean_confidence_before",
                "mean_confidence_after",
                "delta_mean_confidence",
                "extracted_words_before",
                "extracted_words_after",
                "delta_extracted_words",
                "ocr_improved",
            ],
        )
        writer.writeheader()
        for filename, data in metrics.items():
            before = data.get("before", {})
            after = data.get("after", {})
            delta = data.get("delta", {})
            writer.writerow(
                {
                    "file": filename,
                    "selected_engine": data.get("selected_engine", "cnn"),
                    "mean_confidence_before": before.get("mean_confidence", ""),
                    "mean_confidence_after": after.get("mean_confidence", ""),
                    "delta_mean_confidence": delta.get("mean_confidence", ""),
                    "extracted_words_before": before.get("extracted_words", ""),
                    "extracted_words_after": after.get("extracted_words", ""),
                    "delta_extracted_words": delta.get("extracted_words", ""),
                    "ocr_improved": data.get("ocr_improved", ""),
                }
            )

    failed_csv = os.path.join(output_dir, "failed_files.csv")
    with open(failed_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "error"])
        writer.writeheader()
        for err in errors:
            writer.writerow({"file": err.get("file", ""), "error": err.get("error", "")})

    metrics_json = os.path.join(output_dir, "per_file_metrics.json")
    write_json(metrics_json, {"metrics": metrics})


def zip_output_folder(output_dir: str, output_zip_path: str) -> None:
    with ZipFile(output_zip_path, "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)


@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Document Cleaner API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/version")
def version() -> Dict[str, Any]:
    return {
        "name": "Document Cleaner API",
        "api_version": APP_VERSION,
        "engine": "DnCNN",
        "default_weight": DEFAULT_WEIGHT_FILE,
        "supported_formats": ["png", "jpg", "jpeg", "zip"],
        "limits": {
            "max_single_upload_mb": MAX_SINGLE_UPLOAD_MB,
            "max_zip_mb": MAX_ZIP_UPLOAD_MB,
            "max_files_per_zip": MAX_FILES_PER_ZIP,
            "max_extracted_total_mb": MAX_EXTRACTED_TOTAL_MB,
            "max_image_pixels": MAX_IMAGE_PIXELS,
        },
        "api_key_required": bool(os.getenv("DOCUMENT_CLEANER_API_KEY")),
    }


@app.post("/process-document/", dependencies=[Depends(require_api_key)])
@app.post("/api/v1/process-document", dependencies=[Depends(require_api_key)])
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_zip: bool = Query(default=False, description="Return a downloadable ZIP instead of JSON/base64 preview."),
):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    start = time.time()
    check_content_length(request, MAX_SINGLE_UPLOAD_BYTES, "Single image")

    job_id, job_dir, input_dir, output_dir = create_job_dirs()
    safe_name = safe_basename(file.filename)
    ext = validate_extension(safe_name, SUPPORTED_IMAGE_EXTENSIONS)
    validate_mime(file, {"image/png", "image/jpeg"})

    temp_input = os.path.join(input_dir, safe_name)

    try:
        contents = await read_upload_limited(file, MAX_SINGLE_UPLOAD_BYTES)
        with open(temp_input, "wb") as handle:
            handle.write(contents)

        original_image = read_grayscale_image_checked(temp_input)
        denoised_image = denoise_with_cnn(model, temp_input)
        outputs = generate_dual_outputs(denoised_image, original_image=original_image, auto_tune=True)

        stem = Path(safe_name).stem
        human_png = os.path.join(output_dir, f"{stem}_cleaned_human.png")
        ocr_png = os.path.join(output_dir, f"{stem}_cleaned_ocr.png")
        human_pdf = os.path.join(output_dir, f"{stem}_human.pdf")
        ocr_pdf = os.path.join(output_dir, f"{stem}_ocr.pdf")

        write_image_and_pdf(outputs["human"], human_png, human_pdf)
        write_image_and_pdf(outputs["ocr"], ocr_png, ocr_pdf)

        manifest = build_run_manifest(
            job_id=job_id,
            selected_weight=DEFAULT_WEIGHT_FILE,
            files_received=1,
            files_processed=1,
            files_failed=0,
            processing_seconds=time.time() - start,
            output_mode="zip" if return_zip else "json",
        )
        write_json(os.path.join(output_dir, "manifest.json"), manifest)

        logger.info(
            "Processed single document",
            extra={"job_id": job_id, "input_filename": safe_name, "processing_seconds": manifest["processing_seconds"]},
        )

        if return_zip:
            output_zip_path = os.path.join(RESULT_ZIPS_DIR, f"{job_id}_cleaned_document.zip")
            zip_output_folder(output_dir, output_zip_path)
            if not os.path.exists(output_zip_path) or os.path.getsize(output_zip_path) == 0:
                raise RuntimeError("Output ZIP was not created or is empty.")
            background_tasks.add_task(cleanup_job_dir, job_dir)
            background_tasks.add_task(cleanup_file, output_zip_path)
            return FileResponse(
                output_zip_path,
                media_type="application/zip",
                filename="cleaned_document.zip",
                headers={"X-Job-ID": job_id},
            )

        response = {
            "message": "Processing complete",
            "job_id": job_id,
            "best_weight": DEFAULT_WEIGHT_FILE,
            "manifest": manifest,
            "outputs": {
                "human": {
                    "image_base64": encode_image_png(outputs["human"]),
                    "image_filename": Path(human_png).name,
                    "pdf_filename": Path(human_pdf).name,
                },
                "ocr": {
                    "image_base64": encode_image_png(outputs["ocr"]),
                    "image_filename": Path(ocr_png).name,
                    "pdf_filename": Path(ocr_pdf).name,
                },
            },
        }
        cleanup_job_dir(job_dir)
        return response

    except HTTPException:
        cleanup_job_dir(job_dir)
        raise
    except Exception as exc:
        cleanup_job_dir(job_dir)
        logger.exception("Single document processing failed", extra={"job_id": job_id, "input_filename": safe_name})
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")


@app.post("/process-batch/", dependencies=[Depends(require_api_key)])
@app.post("/api/v1/process-batch", dependencies=[Depends(require_api_key)])
async def process_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    start = time.time()
    check_content_length(request, MAX_ZIP_UPLOAD_BYTES, "ZIP")

    job_id, job_dir, input_dir, output_dir = create_job_dirs()
    safe_name = safe_basename(file.filename, fallback="batch.zip")
    validate_extension(safe_name, {".zip"})
    validate_mime(file, {"application/zip", "application/x-zip-compressed"})

    input_zip_path = os.path.join(job_dir, safe_name)
    output_zip_path = os.path.join(RESULT_ZIPS_DIR, f"{job_id}_cleaned_docs.zip")

    try:
        contents = await read_upload_limited(file, MAX_ZIP_UPLOAD_BYTES)
        with open(input_zip_path, "wb") as handle:
            handle.write(contents)

        extracted = safe_extract_zip_images(input_zip_path, input_dir)

        with open(os.path.join(input_dir, "_zip_validation_failures.json"), "r", encoding="utf-8") as handle:
            zip_validation = json.load(handle)
        validation_failures = zip_validation.get("failures", [])

        image_files = sorted((item.get("input_filename") or item.get("filename") or Path(item["path"]).name) for item in extracted)
        sample_percent = 0.2
        min_samples = 3
        max_samples = 10
        num_samples = min(
            max(min_samples, int(len(image_files) * sample_percent)),
            max_samples,
            len(image_files),
        )
        sampled_images = image_files[:num_samples]

        logger.info(
            "Sampling batch for weight selection",
            extra={"job_id": job_id, "sampled": num_samples, "total_images": len(image_files)},
        )

        weight_votes: List[str] = []
        for img_file in sampled_images:
            img_path = os.path.join(input_dir, img_file)
            weight_file = auto_select_best_weight(MODEL_WEIGHTS_DIR, img_path)
            weight_votes.append(weight_file)
            logger.info("Weight vote", extra={"job_id": job_id, "input_filename": img_file, "weight": weight_file})

        best_weight_file = max(set(weight_votes), key=weight_votes.count)

        result = batch_clean_documents(
            weights_path=os.path.join(MODEL_WEIGHTS_DIR, best_weight_file),
            input_folder=input_dir,
            output_folder=output_dir,
            auto_tune=True,
            make_dual_output=True,
        )

        processed = result.get("processed", [])
        failed = [{"file": name, "error": err} for name, err in result.get("failed", [])]
        all_failures = validation_failures + failed

        if not processed:
            raise HTTPException(
                status_code=422,
                detail="Batch ran, but no files were successfully processed. Check failed_files.csv details.",
            )

        top_manifest_path = os.path.join(output_dir, "api_manifest.json")
        api_manifest = build_run_manifest(
            job_id=job_id,
            selected_weight=best_weight_file,
            files_received=len(extracted) + len(validation_failures),
            files_processed=len(processed),
            files_failed=len(all_failures),
            processing_seconds=time.time() - start,
            output_mode="zip",
            failures=all_failures,
        )
        api_manifest["sampling"] = {
            "sample_percent": sample_percent,
            "max_samples": max_samples,
            "num_samples": num_samples,
            "sampled_images": sampled_images,
            "weight_votes": weight_votes,
        }
        write_json(top_manifest_path, api_manifest)

        write_summary_files(output_dir, result.get("manifest"), all_failures)
        zip_output_folder(output_dir, output_zip_path)

        if not os.path.exists(output_zip_path) or os.path.getsize(output_zip_path) == 0:
            raise HTTPException(status_code=500, detail="ZIP not created or empty.")

        background_tasks.add_task(cleanup_job_dir, job_dir)
        background_tasks.add_task(cleanup_file, output_zip_path)

        result_note = (
            f"Sampled {num_samples}/{len(image_files)} images. "
            f"Using shared weight: {best_weight_file}. "
            f"Processed {len(processed)} files; failed {len(all_failures)}."
        )

        logger.info(
            "Processed batch",
            extra={
                "job_id": job_id,
                "files_processed": len(processed),
                "files_failed": len(all_failures),
                "selected_weight": best_weight_file,
                "processing_seconds": api_manifest["processing_seconds"],
            },
        )

        return FileResponse(
            output_zip_path,
            media_type="application/zip",
            filename="cleaned_docs.zip",
            headers={"X-Note": result_note, "X-Job-ID": job_id},
        )

    except HTTPException:
        cleanup_job_dir(job_dir)
        raise
    except Exception as exc:
        cleanup_job_dir(job_dir)
        logger.exception("Batch processing failed", extra={"job_id": job_id, "input_filename": safe_name})
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")

