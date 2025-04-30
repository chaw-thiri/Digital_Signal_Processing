from dotenv import load_dotenv
import os
import re
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

load_dotenv()
API_KEY = os.getenv("OCR_SPACE_API_KEY")

# Font setup (make sure the font exists)
FONT_PATH = os.path.join(os.path.dirname(__file__), "Arial Unicode.ttf")
TINT_COLOR = (255, 255, 0)
TRANSPARENCY = 0.7
OPACITY = int(255 * TRANSPARENCY)

def compress_image(img_path):
    """Compress and resize image to save memory"""
    img = Image.open(img_path)
    img = img.convert('RGB')
    img.thumbnail((1000, 1000))  # Resize to 1000x1000 max
    temp = BytesIO()
    img.save(temp, format='JPEG', quality=85)
    temp.seek(0)
    return temp

def ocr_space_file(filename, api_key=API_KEY, overlay=True, language="eng", engine=2):
    """Send image to OCR.space API"""
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
        'OCREngine': engine,
        'detectOrientation': True,
        'scale': True,
    }
    f = compress_image(filename)
    response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'file': ('image.jpg', f)},
        data=payload
    )
    return response.json()

def text_extraction(image_path, ocr_data):
    """Extract text and create overlay"""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new('RGBA', img.size, TINT_COLOR + (0,))
    draw = ImageDraw.Draw(overlay)

    clientId = None
    package_weight = None

    for line in ocr_data[0]['TextOverlay']['Lines']:
        for word in line['Words']:
            x1 = (word["Left"], word["Top"])
            x2 = (x1[0] + word["Width"], x1[1] + word["Height"])
            font_size = abs(x1[1] - x2[1]) or 12
            try:
                font = ImageFont.truetype(FONT_PATH, font_size)
            except:
                font = ImageFont.load_default()

            draw.rectangle((x1, x2), fill=TINT_COLOR + (OPACITY,))
            draw.text(x1, word["WordText"], fill=(255, 0, 0, 255), font=font)

            if not clientId:
                match = re.findall(r'\bamk\d{5}\b', word["WordText"].lower())
                if match:
                    clientId = match[0]
            if not package_weight:
                match = re.findall(r'\b\d+(?:\.\d+)?\s*kg\b', word["WordText"].lower())
                if match:
                    package_weight = match[0]

    img = Image.alpha_composite(img, overlay)
    # Save in a results directory
    results_dir = os.path.join(os.path.dirname(image_path), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_overlay.png")
    img.convert("RGB").save(output_path)
    print(f"[✔] Saved: {output_path}")

    if clientId and package_weight:
        print(f"[📦] ID: {clientId} | Weight: {package_weight}")
    elif clientId:
        print(f"[📦] ID: {clientId} | No weight found.")
    elif package_weight:
        print(f"[📦] Weight: {package_weight} | No ID found.")
    else:
        print("[❌] No ID or weight detected.")

def run_single_img_ocr(img_path):
    """Process a single image"""
    result = ocr_space_file(img_path)
    if not result.get("IsErroredOnProcessing", True):
        parsed = result.get("ParsedResults")
        if parsed:
            text_extraction(img_path, parsed)
        else:
            print("[!] No text found.")
    else:
        print("[!] OCR Error:", result.get("ErrorMessage", "Unknown error")) 