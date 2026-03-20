import json
import base64

with open("response.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

outputs = obj.get("outputs", {})

human_b64 = outputs.get("human", {}).get("image_base64")
ocr_b64 = outputs.get("ocr", {}).get("image_base64")

if human_b64:
    with open("cleaned_human.png", "wb") as f:
        f.write(base64.b64decode(human_b64))
    print("Saved cleaned_human.png")
else:
    print("No human image found in response.json")

if ocr_b64:
    with open("cleaned_ocr.png", "wb") as f:
        f.write(base64.b64decode(ocr_b64))
    print("Saved cleaned_ocr.png")
else:
    print("No ocr image found in response.json")