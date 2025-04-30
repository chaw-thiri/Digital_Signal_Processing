import cv2
from pyzbar.pyzbar import decode
from PIL import Image

def detect_barcodes(image_path):
    # Read the image using OpenCV
    img_cv2 = cv2.imread(image_path)
    
    if img_cv2 is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Convert BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_rgb)

    # Decode barcodes using pyzbar
    barcodes = decode(img_pil)

    if not barcodes:
        print("No barcodes found.")
    else:
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            print(f"[+] Found {barcode_type} barcode: {barcode_data}")
