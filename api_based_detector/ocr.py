# from dotenv import load_dotenv
import os
import re
import requests
from PIL import Image, ImageDraw
import json

# # Load environment variables
# load_dotenv()
API_KEY = os.getenv("OCR_API_KEY") or "K83105096488957"

# Parameters for text overlay 
TINT_COLOR = (255, 255, 0)  # Yellow highlight
TRANSPARENCY = 0.70
OPACITY = int(255 * TRANSPARENCY)
TEXT_COLOR = (255, 0, 0)  # Red text

def text_extraction(image_path, ocr_data):
    """Extract text from OCR data and overlay it on the image"""
    # Read the image
    img = Image.open(image_path)
    img = img.convert("RGBA")

    # Create an overlay for highlighting
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    client_id = None
    package_weight = None
    
    # Iterate over the OCR data to extract word bounding boxes
    for line in ocr_data[0]['TextOverlay']['Lines']:
        for word in line['Words']:
            # Get word coordinates
            left = word["Left"]
            top = word["Top"]
            width = word["Width"]
            height = word["Height"]
            
            # Draw rectangle for highlighting
            draw.rectangle(
                [(left, top), (left + width, top + height)],
                fill=TINT_COLOR + (OPACITY,)
            )
            
            # No need to draw text - we'll skip that to avoid font issues
            
            # Extract detection information
            detected_text = word["WordText"]
            
            # Look for client ID (amk followed by 5 digits)
            client_id_matches = re.findall(r'\bamk\d{5}\b', detected_text.lower())
            if client_id_matches:
                client_id = client_id_matches[0]
                
            # Look for package weight (number followed by kg)
            package_weight_matches = re.findall(r'\b\d+(?:\.\d+)?\s*kg\b', detected_text.lower())
            if package_weight_matches:
                package_weight = package_weight_matches[0]

    # Print the extracted information
    if client_id and package_weight:
        print(f"ID: {client_id}, Package weight: {package_weight}")
    elif client_id:
        print(f"ID: {client_id}, No weight detected")
    elif package_weight:
        print(f"Package weight: {package_weight}, No ID present")
    else: 
        print("No ID or weight detected")
            
    # Combine the original image with the overlay
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")  # Convert back to RGB for saving
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(image_path), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed image
    base_filename = os.path.basename(image_path)
    output_file_name = os.path.join(output_dir, f"processed_{base_filename}")
    img.save(output_file_name)
    print(f"Processed image saved as: {output_file_name}")
    
    # Show the image (this might not work in all environments)
    try:
        img.show()
    except Exception as e:
        print(f"Could not display image: {e}")

def ocr_space_file(filename, api_key=API_KEY, overlay=True, language="auto", engine=2):
    """OCR.space API request with local file"""
    # If no API key is provided, use demo key
    if not api_key:
        api_key = "helloworld"  # Default demo key
        print("Warning: Using demo API key. For better results, set OCR_SPACE_API_KEY in .env file")
        
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
        'OCREngine': engine,
        'detectOrientation': True, 
        'scale': True,
    }
    
    try:
        with open(filename, 'rb') as f:
            r = requests.post(
                'https://api.ocr.space/parse/image',
                files={'file': f},
                data=payload,
            )
        return r.json()
    except Exception as e:
        print(f"Error in OCR API request: {e}")
        return {"IsErroredOnProcessing": True, "ErrorMessage": str(e)}

def run_single_img_ocr(img_path):
    """Process a single image with OCR"""
    print(f"Processing image: {img_path}")
    
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"Error: File not found: {img_path}")
        return
        
    # Send to OCR API
    result = ocr_space_file(img_path)

    # Check if there is no error
    if result.get("IsErroredOnProcessing") == False:
        parsed_results = result.get("ParsedResults")
        
        if parsed_results and len(parsed_results) > 0:
            # Check if TextOverlay exists
            if 'TextOverlay' in parsed_results[0] and 'Lines' in parsed_results[0]['TextOverlay']:
                text_extraction(img_path, parsed_results)
            else:
                print("OCR completed but no text overlay found in response")
                print(f"Response structure: {list(parsed_results[0].keys())}")
        else:
            print("No text found in the image.")
    else:
        print("OCR Error:", result.get("ErrorMessage", "Unknown error occurred."))
        
    print("----------------------------")