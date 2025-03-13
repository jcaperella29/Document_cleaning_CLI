# 📄 Document Cleaner API 🧼🧠

A high-performance Python-based system for **denoising scanned documents**, **enhancing text clarity**, and exporting **OCR-optimized PDFs** using deep learning and image post-processing.

Supports both **Command-Line Interface (CLI)** and **REST API** usage.  
Built with: **PyTorch**, **OpenCV**, **FastAPI**, **Tesseract OCR**

---

## 🚀 What Can You Do With It?

- Upload scanned documents (.jpg/.png)
- Automatically clean & denoise them using AI
- Get cleaned images + searchable PDFs back
- Use it from terminal or as a web API
- Deploy it locally or to the cloud

---

## 🔧 Quick Setup

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/document-cleaner.git
cd document-cleaner

###set up virtual envirment 

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

#Install Dependencies
pip install -r requirements.txt

---

## 📸 Install Tesseract OCR

This app uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) to extract text from images and optimize model performance.

You **must install it separately** on your system — it's not included in pip packages.

---

### 🐧 Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install tesseract-ocr


 macOS (using Homebrew)
bash
Copy
Edit
brew install tesseract

🪟 Windows
Download installer from:
👉 https://github.com/UB-Mannheim/tesseract/wiki

Run the .exe and install to:

makefile
Copy
Edit
C:\Program Files\Tesseract-OCR
Add Tesseract to your PATH:

Open Start Menu → search Environment Variables
Edit the PATH variable
Add:
makefile
Copy
Edit
C:\Program Files\Tesseract-OCR
Reboot terminal (or restart system) and run:

bash
Copy
Edit
tesseract --version


🧑‍💻 Option 1: CLI (Command Line Interface)
Clean a batch of images directly from terminal.

🪄 Step-by-step:
Place .mat model files inside a folder (e.g. model_weights/)
Put your noisy images into a folder (e.g. input_docs/)
Run the script:
bash
Copy
Edit
python processor.py model_weights/ input_docs/ output_docs/ --auto-tune --auto-select
✅ This will:

Select the best .mat model based on OCR quality
Clean every image
Save both cleaned .png and .pdf to output_docs/
🌐 Option 2: API Mode (Run Locally)
Start a web API using FastAPI:

bash
Copy
Edit
uvicorn api:app --reload
Visit:
📍 http://localhost:8000/docs (Swagger UI)

💡 Use the API:
🔹 Clean a Single Image
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/process-document/' \
  -F 'file=@sample.png' \
  -H 'accept: application/json'
✅ Returns:

cleaned_image_base64: preview of cleaned PNG
download_pdf: endpoint to download final PDF
🔹 Clean a Zip of Images
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/process-batch/' \
  -F 'file=@your_batch.zip' \
  -H 'accept: application/zip' \
  --output cleaned_output.zip
✅ Returns:

A zip file containing all cleaned .pdf results
🔹 Download PDF
After uploading a file, you’ll get a URL like:

bash
Copy
Edit
http://localhost:8000/download/sample.pdf
You can access that link to get the cleaned PDF.




⚙️ How to Use
🧑‍💻 Option 1: CLI (Command Line Interface)
Clean a batch of images directly from terminal.

🪄 Step-by-step:
Place .mat model files inside a folder (e.g. model_weights/)
Put your noisy images into a folder (e.g. input_docs/)
Run the script:
bash
Copy
Edit
python processor.py model_weights/ input_docs/ output_docs/ --auto-tune --auto-select
✅ This will:

Select the best .mat model based on OCR quality
Clean every image
Save both cleaned .png and .pdf to output_docs/

🌐 Option 2: API Mode (Run Locally)
Start a web API using FastAPI:

bash
Copy
Edit
uvicorn api:app --reload
Visit:
📍 http://localhost:8000/docs (Swagger UI)

💡 Use the API:
🔹 Clean a Single Image
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/process-document/' \
  -F 'file=@sample.png' \
  -H 'accept: application/json'
✅ Returns:

cleaned_image_base64: preview of cleaned PNG
download_pdf: endpoint to download final PDF
🔹 Clean a Zip of Images
bash
Copy
Edit
curl -X 'POST' \
  'http://localhost:8000/process-batch/' \
  -F 'file=@your_batch.zip' \
  -H 'accept: application/zip' \
  --output cleaned_output.zip
✅ Returns:

A zip file containing all cleaned .pdf results
🔹 Download PDF
After uploading a file, you’ll get a URL like:

bash
Copy
Edit
http://localhost:8000/download/sample.pdf
You can access that link to get the cleaned PDF.


.
├── main.py                 # FastAPI API logic
├── processor.py           # DnCNN + OCR processing logic
├── pubsub_client.py       # (Optional) Pub/Sub for async mode
├── worker.py              # (Optional) background processor
├── model_weights/         # Your .mat weight files go here
├── uploads/               # Where uploaded files go
├── processed/             # Output cleaned .png and .pdf
├── requirements.txt
└── README.md

