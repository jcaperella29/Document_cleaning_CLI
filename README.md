# 🖼️ Document Cleaning Pipeline 🚀

This project is a **document image cleaning pipeline** using **deep learning-based denoising and OCR optimization**.

## ✨ Features
✔ Auto-selects the best noise-removal model  
✔ Auto-tunes post-processing parameters for best OCR results  
✔ Converts cleaned images into high-quality PDFs  
✔ Includes **sample images** for testing  
✔ Command-line interface for easy use  

---

## 📥 Installation

### 1️⃣ Clone the repository
```sh
git clone https://github.com/YOUR-USERNAME/document-cleaning-pipeline.git
cd document-cleaning-pipeline

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Ensure Tesseract-OCR is installed
Download & install Tesseract-OCR
Then set the correct path in Document_cleaning_CLI.py:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

 Usage
✅ Run Auto-Tuned Document Cleaning
s
python Document_cleaning_CLI.py "model_weights" "sample_data_for_testing" "output_docs" --auto-select --auto-tune
✅ Run with Manual Parametersh


python Document_cleaning_CLI.py "model_weights" "sample_data_for_testing" "output_docs" --sharpen --blend-factor 0.3
✅ Example Output
Input: Noisy scanned document
Output: Cleaned, readable document ready for OCR and PDF conversion
🖼 Sample Images
This repository includes a collection of sample scanned document images inside the sample_data_for_testing folder.

📌 You can test the script using these sample images.


python Document_cleaning_CLI.py "model_weights" "sample_data_for_testing" "output_docs" --auto-tune
✅ This will process the sample images and generate cleaned PDFs!

📜 License
This project is open-source and available under the MIT License.
Feel free to contribute! 🎉
