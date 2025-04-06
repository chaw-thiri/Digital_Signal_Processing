import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

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
        
        # Save the preprocessed image for inspection (don’t delete it yet)
        temp_image_path = "temp_processed_image.png"
        cv2.imwrite(temp_image_path, processed_img)
        print(f"Preprocessed image saved as: {temp_image_path}")
        
        # Configure Tesseract for Korean and English with PSM 6
        custom_config = r'--oem 3 --psm 6 -l kor+eng'
        
        # Perform OCR
        text = pytesseract.image_to_string(
            Image.open(temp_image_path),
            config=custom_config
        )
        
        # Get detailed data including confidence scores
        details = pytesseract.image_to_data(
            Image.open(temp_image_path),
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Comment out removal so you can inspect the image
        # os.remove(temp_image_path)
        
        return text, details
        
    except Exception as e:
        return f"Error: {str(e)}", None

def main():
    image_path = "/Users/shukurullomeliboyev2004/Desktop/university/Digital_Signal_Processing/test1.png"
    if not os.path.exists(image_path):
        print("Image file not found!")
        return
    
    text, details = ocr_image(image_path)
    print("Extracted Text:")
    print(text)
    if details:
        print("\nDetailed Results:")
        n_boxes = len(details['text'])
        for i in range(n_boxes):
            if int(float(details['conf'][i])) > 60:
                print(f"Text: {details['text'][i]}, Confidence: {details['conf'][i]}%")

if __name__ == "__main__":
    main()