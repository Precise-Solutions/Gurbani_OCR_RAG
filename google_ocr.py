from google.cloud import vision
import os

# Tell Google where your service account key is
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/parmeetsingh/Desktop/Gurbani Cleaning/OCR_key.json"

def extract_text(img_path):
    client = vision.ImageAnnotatorClient()

    with open(img_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        print("API ERROR:", response.error.message)
        return

    text = response.text_annotations[0].description if response.text_annotations else ""
    print("\n=== OCR RESULT ===\n")
    print(text)

# FULL path to your image
extract_text("/Users/parmeetsingh/Desktop/Gurbani Cleaning/Page_4.jpeg")

