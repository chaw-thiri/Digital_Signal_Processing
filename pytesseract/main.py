import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
import glob

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to make text larger (2x scaling)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply Otsu's thresholding for automatic binarization
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def ocr_image(image_path):
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Configure Tesseract for Korean and English with PSM 6
        custom_config = r'--oem 3 --psm 6 -l kor+eng'
        
        # Perform OCR
        text = pytesseract.image_to_string(
            Image.fromarray(processed_img),
            config=custom_config
        )
        
        # Get detailed data including confidence scores
        details = pytesseract.image_to_data(
            Image.fromarray(processed_img),
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        return text, details
        
    except Exception as e:
        return f"Error: {str(e)}", None

def process_all_images():
    # Get all image files from test_images directory
    image_files = glob.glob(os.path.join("test_images", "*.png"))
    image_files.extend(glob.glob(os.path.join("test_images", "*.jpg")))
    image_files.extend(glob.glob(os.path.join("test_images", "*.jpeg")))
    
    if not image_files:
        print("No image files found in test_images directory!")
        return
    
    print("\n=== OCR Results ===\n")
    
    for image_path in sorted(image_files):
        image_name = os.path.basename(image_path)
        text, details = ocr_image(image_path)
        
        print(f"{image_name} extracted text:")
        print(text.strip())
        
        if details:
            print("\nDetailed Results:")
            n_boxes = len(details['text'])
            for i in range(n_boxes):
                if int(float(details['conf'][i])) > 60:
                    print(f"Text: {details['text'][i]}, Confidence: {details['conf'][i]}%")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    process_all_images()