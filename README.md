<p align="center">
  <img src="https://img.shields.io/badge/Cloud%20Run-Ready-blueviolet?logo=googlecloud&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-%F0%9F%9A%80-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Auto%20Tuning-Enabled-success?logo=ai" />
  <img src="https://img.shields.io/badge/API%20Hardening-v0.2.4-success" />
</p>

# 📄 Document Cleaner API 🧼🧠

A Python document-cleaning system for **denoising scanned document images**, **enhancing text clarity**, and exporting both **human-readable** and **OCR-optimized** outputs using deep learning + image processing.

Supports both:

- **Command-Line Interface (CLI)** workflows
- **FastAPI REST API** workflows

Built with **PyTorch**, **OpenCV**, **FastAPI**, and **Tesseract OCR**.

---

## 🚀 What Can You Do With It?

- Upload scanned documents as `.png`, `.jpg`, `.jpeg`, or a ZIP of images.
- Automatically clean and denoise images using a DnCNN-based model.
- Generate separate:
  - human-readable cleaned PNGs/PDFs
  - OCR-optimized cleaned PNGs/PDFs
- Process single images or batch ZIP uploads.
- Auto-select a denoising weight for batch jobs using OCR-guided sampling.
- Receive audit artifacts such as manifests, summary CSVs, failed-file reports, and per-file OCR metrics.
- Run locally, through the CLI, or as a REST API suitable for private/cloud deployment.

---

## 🆕 Production Hardening Upgrades

The API wrapper has been hardened for safer demo and deployment usage.

### Security and upload safety

- Optional API-key gate using the `X-API-Key` header.
- Upload size limits for single images and ZIP files.
- Maximum files-per-ZIP limit.
- Maximum extracted ZIP size limit.
- Maximum image pixel limit.
- Supported input validation for `.png`, `.jpg`, `.jpeg`, and `.zip`.
- Safer ZIP handling that rejects:
  - invalid ZIP files
  - path traversal entries such as `../`
  - absolute paths
  - hidden file/folder paths
  - unsupported file types

### Batch reliability

- Bad files inside a ZIP no longer kill the whole batch.
- Valid images are processed.
- Unsupported or unreadable files are recorded in `failed_files.csv`.
- Job directories are cleaned up after responses.
- Stale job cleanup runs at API startup.

### Product/audit artifacts

Batch output ZIPs now include:

```text
api_manifest.json
manifest.json
summary.csv
failed_files.csv
per_file_metrics.json
*_cleaned_human.png
*_cleaned_ocr.png
*_human.pdf
*_ocr.pdf
```

This makes the output easier to inspect, reproduce, and evaluate.

---

## 🧠 Weight Selection Logic

When a ZIP is uploaded, the API performs OCR-guided adaptive sampling.

Current behavior:

- Samples **20% of images**
- Uses a minimum of **3 samples when possible**
- Caps sampling at **10 images**
- Runs `auto_select_best_weight(...)` on each sampled image
- Picks the most common winning weight
- Applies that shared weight to the full batch

Expected sampling behavior:

```text
1-image batch   -> samples 1
2-image batch   -> samples 2
5-image batch   -> samples 3
20-image batch  -> samples 4
100-image batch -> samples 10
```

Example `api_manifest.json` sampling block:

```json
{
  "sample_percent": 0.2,
  "max_samples": 10,
  "num_samples": 3,
  "sampled_images": [
    "noisy_0.png",
    "noisy_1.png",
    "noisy_2.png"
  ],
  "weight_votes": [
    "sigma=60.mat",
    "sigma=45.mat",
    "sigma=70.mat"
  ]
}
```

---

## 🌐 API Endpoints

### Health check

```bash
curl http://localhost:8080/health
```

Expected:

```json
{"status":"ok"}
```

### Version/about endpoint

```bash
curl http://localhost:8080/api/v1/version
```

Example response:

```json
{
  "name": "Document Cleaner API",
  "api_version": "0.2.4",
  "engine": "DnCNN",
  "default_weight": "sigma=20.mat",
  "supported_formats": ["png", "jpg", "jpeg", "zip"],
  "api_key_required": true
}
```

### Clean a single image

JSON/base64 preview mode:

```bash
curl -X POST \
  -H "X-API-Key: change-me-before-demo" \
  -F "file=@sample.png" \
  http://localhost:8080/api/v1/process-document
```

Downloadable ZIP mode:

```bash
curl -X POST \
  -H "X-API-Key: change-me-before-demo" \
  -F "file=@sample.png" \
  "http://localhost:8080/api/v1/process-document?return_zip=true" \
  --output cleaned_document.zip
```

### Clean a ZIP batch

```bash
curl -f -S -D response_headers.txt -X POST \
  -H "X-API-Key: change-me-before-demo" \
  -F "file=@real_batch_test.zip" \
  http://localhost:8080/api/v1/process-batch \
  --output cleaned_docs.zip
```

Inspect output:

```bash
rm -rf cleaned_test_output
mkdir -p cleaned_test_output
unzip -o cleaned_docs.zip -d cleaned_test_output
ls -lah cleaned_test_output
cat cleaned_test_output/api_manifest.json
cat cleaned_test_output/summary.csv
cat cleaned_test_output/failed_files.csv
```

---

## 🔐 API Key Protection

Set an API key before running the server:

```bash
export DOCUMENT_CLEANER_API_KEY="change-me-before-demo"
uvicorn main:app --host 0.0.0.0 --port 8080
```

Then requests to processing endpoints must include:

```bash
-H "X-API-Key: change-me-before-demo"
```

If the key is missing or invalid, the API returns:

```json
{"detail":"Missing or invalid API key."}
```

If `DOCUMENT_CLEANER_API_KEY` is not set, the API remains open for local development.

---

## ✅ Hardening Smoke Tests

### 1. Confirm API key is active

```bash
curl http://localhost:8080/api/v1/version
```

Look for:

```json
"api_key_required": true
```

### 2. Missing API key should fail

```bash
curl -i -X POST \
  -F "file=@real_batch_test.zip" \
  http://localhost:8080/api/v1/process-batch
```

Expected:

```text
HTTP/1.1 401 Unauthorized
```

### 3. Invalid ZIP should fail cleanly

```bash
echo "not a real zip" > fake.zip

curl -i -X POST \
  -H "X-API-Key: change-me-before-demo" \
  -F "file=@fake.zip;type=application/zip" \
  http://localhost:8080/api/v1/process-batch
```

Expected:

```json
{"detail":"Uploaded file is not a valid ZIP archive."}
```

### 4. Mixed ZIP should process good files and report bad files

```bash
mkdir -p mixed_bad_batch
cp noisy_0.png mixed_bad_batch/
echo "hello" > mixed_bad_batch/not_image.txt
zip -r mixed_bad_batch.zip mixed_bad_batch

curl -f -S -D response_headers.txt -X POST \
  -H "X-API-Key: change-me-before-demo" \
  -F "file=@mixed_bad_batch.zip" \
  http://localhost:8080/api/v1/process-batch \
  --output mixed_cleaned_docs.zip

rm -rf mixed_cleaned_output
mkdir -p mixed_cleaned_output
unzip -o mixed_cleaned_docs.zip -d mixed_cleaned_output
cat mixed_cleaned_output/failed_files.csv
```

Expected `failed_files.csv` example:

```csv
file,error
mixed_bad_batch/not_image.txt,Skipped unsupported file type.
```

---

## 🔧 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/jcaperella29/Document_cleaning_CLI.git
cd Document_cleaning_CLI
```

### 2. Install Tesseract OCR

Linux/WSL:

```bash
sudo apt update && sudo apt install tesseract-ocr
```

macOS:

```bash
brew install tesseract
```

Windows:

Download Tesseract from the UB Mannheim installer page and add it to your system PATH:

```text
C:\Program Files\Tesseract-OCR\
```

Verify:

```bash
tesseract --version
```

If needed, set the explicit path:

```bash
export TESSERACT_CMD="/usr/bin/tesseract"
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Confirm model weights

Place `.mat` DnCNN weight files in:

```text
model_weights/
```

Example:

```text
model_weights/sigma=20.mat
model_weights/sigma=40.mat
model_weights/sigma=45.mat
model_weights/sigma=60.mat
model_weights/sigma=70.mat
```

---

## ⚙️ Usage Options

### Option 1: CLI

```bash
python processor.py model_weights/ input_docs/ output_docs/ --auto-tune --auto-select
```

Useful CLI flags:

```bash
--auto-tune       # OCR-aware post-processing parameter tuning
--auto-select     # select best CNN weight from model_weights/
--single-output   # only save human-readable output
--engine cnn      # use CNN engine
--engine sbb      # use SBB binarization engine, if configured
--engine auto     # compare CNN vs SBB per file, if SBB is configured
```

### Option 2: Local API Mode

```bash
export DOCUMENT_CLEANER_API_KEY="change-me-before-demo"
uvicorn main:app --host 0.0.0.0 --port 8080
```

Open Swagger docs:

```text
http://localhost:8080/docs
```

---

## ⚙️ Environment Variables

```bash
export DOCUMENT_CLEANER_API_KEY="change-me-before-demo"
export APP_VERSION="0.2.4"
export MODEL_WEIGHTS_DIR="model_weights"
export DEFAULT_WEIGHT_FILE="sigma=20.mat"

export MAX_SINGLE_UPLOAD_MB="25"
export MAX_ZIP_UPLOAD_MB="250"
export MAX_FILES_PER_ZIP="100"
export MAX_EXTRACTED_TOTAL_MB="500"
export MAX_IMAGE_PIXELS="25000000"
export JOB_TTL_HOURS="12"
```

---

## 📁 Project Structure

```text
.
├── main.py                 # Hardened FastAPI API wrapper
├── processor.py            # DnCNN + OCR processing logic
├── docclean/
│   ├── manifest.py         # Manifest helpers
│   └── engines/
│       └── sbb.py          # Optional SBB integration
├── model_weights/          # .mat model weights
├── input_docs/             # Example input folder
├── output_docs/            # Example output folder
├── requirements.txt
└── README.md
```

---

## 🔁 Integration Examples

### Python integration

```python
import requests

API_URL = "http://localhost:8080/api/v1/process-batch"
API_KEY = "change-me-before-demo"

def clean_zip(zip_path: str, output_path: str = "cleaned_docs.zip") -> None:
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path, f, "application/zip")}
        headers = {"X-API-Key": API_KEY}
        response = requests.post(API_URL, files=files, headers=headers, timeout=600)

    if response.status_code == 200:
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"Cleaned ZIP saved to {output_path}")
    else:
        print("Error:", response.status_code, response.text)

clean_zip("real_batch_test.zip")
```

### Shell integration

```bash
#!/bin/bash

API_URL="http://localhost:8080/api/v1/process-batch"
API_KEY="change-me-before-demo"

for zip in ./zips/*.zip; do
  echo "Cleaning $zip..."
  curl -f -S -X POST \
    -H "X-API-Key: $API_KEY" \
    -F "file=@$zip" \
    "$API_URL" \
    --output "cleaned_$(basename "$zip")"
done
```

### JavaScript integration

```javascript
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

async function cleanZip(zipPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(zipPath));

  const response = await axios.post(
    "http://localhost:8080/api/v1/process-batch",
    form,
    {
      headers: {
        ...form.getHeaders(),
        "X-API-Key": "change-me-before-demo",
      },
      responseType: "stream",
      timeout: 600000,
    }
  );

  const output = fs.createWriteStream("cleaned_docs.zip");
  response.data.pipe(output);

  console.log("Cleaned ZIP downloaded.");
}

cleanZip("./uploads/images_batch.zip");
```

Install JS dependencies:

```bash
npm install axios form-data
```

---

## 📦 Output Bundle

A successful batch returns `cleaned_docs.zip`.

Example contents:

```text
api_manifest.json
manifest.json
summary.csv
failed_files.csv
per_file_metrics.json
noisy_0_cleaned_human.png
noisy_0_cleaned_ocr.png
noisy_0_human.pdf
noisy_0_ocr.pdf
...
```

### `api_manifest.json`

High-level API run metadata:

- job ID
- API version
- selected DnCNN weight
- files received/processed/failed
- processing time
- input limits
- sampled images
- weight votes
- failure list

### `summary.csv`

Compact per-file OCR metric summary.

### `failed_files.csv`

Files rejected or failed during processing.

Example:

```csv
file,error
mixed_bad_batch/not_image.txt,Skipped unsupported file type.
```

### `per_file_metrics.json`

Machine-readable per-file OCR metrics, including before/after confidence, extracted words, extracted characters, and OCR improvement flags.

---

## 💡 Use Cases

- OCR preprocessing for noisy scanned forms
- Archival cleanup workflows
- Batch cleaning of historical documents
- Internal document-processing APIs
- Preprocessing before downstream OCR, NLP, or document indexing
- Reproducible document-cleaning pipelines with audit artifacts

---

## ⚠️ Current Limitations

- Inputs are image files only: `.png`, `.jpg`, `.jpeg`.
- ZIP uploads should contain image files, not PDFs.
- PDF input parsing is not currently supported.
- OCR improvement is measured using Tesseract-derived metrics and should be validated on your target document type.
- Long-running large batches may eventually benefit from async job submission, polling, and signed download URLs.

---

## ✅ Final Notes

The core engine performs DnCNN-based cleaning and OCR-aware post-processing. The FastAPI layer now adds the deployment-facing wrapper pieces needed for a more serious demo:

- safer uploads
- safer ZIP handling
- API-key access control
- automatic cleanup
- manifest and summary artifacts
- batch-level weight selection
- per-file failure reporting

That makes the project easier to evaluate, easier to demo, and closer to a production-style document-cleaning workflow API.





