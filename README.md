# 📄 Document Cleaner API 🧼🧠

A high-performance Python system for **denoising scanned documents**, **enhancing text clarity**, and exporting **OCR-optimized PDFs** using deep learning + image processing.

Supports both **Command-Line Interface (CLI)** and **REST API** usage.  
Built with: **PyTorch**, **OpenCV**, **FastAPI**, **Tesseract OCR**

---

## 🚀 What Can You Do With It?

- Upload scanned documents (.jpg/.png or a ZIP of images)
- Automatically clean & denoise them using deep learning
- Get back cleaned `.png` images and OCR-ready `.pdf` files
- Use it via terminal (CLI) or as a live REST API
- ✅ Now supports **cloud deployments** (e.g. GCP Cloud Run)

---

## 🌐 Live API (Cloud Run Deployment)

Base URL:

https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/ 


### 🔹 Clean a Single Image

```bash
curl -X POST \
  -F "file=@sample.png" \
  https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/process-document/

### Clean a ZIP of Images

curl -X POST \
  -F "file=@your_batch.zip" \
  https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/process-batch/ \
  --output cleaned_output.zip



###🧠 Weight Selection Logic (NEW)
🆕 Adaptive Sampling Strategy:
When a ZIP is uploaded, the app:
Samples 20% of the total images, capped at 10 images
Runs auto_select_best_weight(...) on each sample
Picks the most common weight (based on OCR accuracy)
✅ Applies that shared best weight to the full batch
This keeps runtime fast even on large batches while preserving per-batch tuning accuracy 💨🎯

###🔧 Local Setup
1. Clone the Repo

git clone https://github.com/your-username/document-cleaner.git
cd document-cleaner

##📸 Install Tesseract OCR
 On 🐧 Linux/Bash run 

sudo apt update && sudo apt install tesseract-ocr

On 🍏 macOS run

brew install tesseract

On 🪟 Windows
Download: https://github.com/UB-Mannheim/tesseract/wiki
Add to: C:\Program Files\Tesseract-OCR\ in your system PATH
Run tesseract --version to verify in powershell/cmd terminal

###⚙️ Usage Options
🧑‍💻 Option 1: CLI (Command Line)

python processor.py model_weights/ input_docs/ output_docs/ --auto-tune --auto-select

✅ Auto-selects best weight for each image
✅ Applies denoising and saves .png and .pdf files

🌐 Option 2: Local API Mode
Start the server:

uvicorn main:app --reload
Open Swagger docs:

http://localhost:8000/docs
Use /process-document/ or /process-batch/ endpoints just like the cloud version.

📁 Project Structure

.
├── main.py                 # FastAPI API logic
├── processor.py           # DnCNN + OCR processing logic
├── model_weights/         # Place your .mat model weights here
├── uploads/               # Temp upload folder
├── processed/             # Output folder for cleaned docs
├── requirements.txt
└── README.md

---

## 🔁 Integration Examples (Python, Shell, JavaScript)

Need to process files in chunks? Automate cleaning in a script? Here's how to integrate it from other tools:

---

### 🐍 Python Integration (using `requests`)

```python
import requests

def clean_zip(zip_path):
    url = "https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/process-batch/"
    with open(zip_path, 'rb') as f:
        files = {'file': (zip_path, f, 'application/zip')}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            with open("cleaned_docs.zip", "wb") as out_file:
                out_file.write(response.content)
            print("✅ Cleaned ZIP saved.")
        else:
            print("❌ Error:", response.status_code, response.text)

🧠 You can wrap this in a loop and call it per batch.


#🐚 Shell Integration (Bash Script)

#!/bin/bash

API_URL="https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/process-batch/"

for zip in ./zips/*.zip; do
  echo "📦 Cleaning $zip..."
  curl -X POST -F "file=@$zip" "$API_URL" --output "cleaned_$(basename "$zip")"
done
✅ Processes all ZIPs in the zips/ folder
🧠 You can run this with: bash batch_clean.sh(assuming you call this script batch_clean.sh)



###🌐 JavaScript Integration (Node.js + Axios)
Useful if you're building a UI, Electron app, or automation tool

const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

async function cleanZip(zipPath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(zipPath));

  const response = await axios.post(
    'https://document-cleaning-cli-111-777-888-7777-934773375188.us-central1.run.app/process-batch/',
    form,
    {
      headers: form.getHeaders(),
      responseType: 'stream'
    }
  );

  const output = fs.createWriteStream('cleaned_docs.zip');
  response.data.pipe(output);

  console.log('✅ Cleaned ZIP downloaded.');
}

cleanZip('./uploads/images_batch.zip');

#📦 Requires:
npm install axios form-data


###💡 Use Case: Chunked Cleaning
These integration examples let you:

Loop through batches of images or ZIPs
Parallelize uploads via API
Hook into your own software (Python backend, JS frontend, shell automation)
📂 Great for large archives, archival pipelines, or headless batch jobs.



✅ Final Notes
🧠 Weight selection is fully automatic (adaptive sampling)
🧼 Output is clean, high-contrast, and PDF-ready
💨 Designed for cloud performance (GCP/Cloud Run ready)







