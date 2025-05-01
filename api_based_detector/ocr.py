from dotenv import load_dotenv
import os
import re
import requests
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OCR_SPACE_API_KEY")

# Parameters for text overlay 
TINT_COLOR = (255, 255, 0)  # Yellow highlight
TRANSPARENCY = 0.70
OPACITY = int(255 * TRANSPARENCY)
TEXT_COLOR = (255, 0, 0)  # Red text

def preprocess_for_seven_segment(image):
    """Apply specialized preprocessing for seven-segment displays"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive threshold to handle lighting variations
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's threshold to find optimal threshold value
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create a combined threshold image for better results
    combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
    
    # Morphological operations to clean up noise and connect segments
    kernel = np.ones((3, 3), np.uint8)
    morph_clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return morph_clean, otsu_thresh, adaptive_thresh, combined

def detect_seven_segment_digits(image_path):
    """Detect and extract seven-segment display values from the image"""
    # Read the image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return None
    
    # Create output directory for debug images
    output_dir = os.path.join(os.path.dirname(image_path), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get different preprocessed versions of the image
    morph_clean, otsu_thresh, adaptive_thresh, combined = preprocess_for_seven_segment(img)
    
    # Save preprocessed images for debugging
    cv2.imwrite(os.path.join(output_dir, f"1_otsu_{os.path.basename(image_path)}"), otsu_thresh)
    cv2.imwrite(os.path.join(output_dir, f"2_adaptive_{os.path.basename(image_path)}"), adaptive_thresh)
    cv2.imwrite(os.path.join(output_dir, f"3_combined_{os.path.basename(image_path)}"), combined)
    cv2.imwrite(os.path.join(output_dir, f"4_morph_{os.path.basename(image_path)}"), morph_clean)
    
    # Find contours on the processed image
    contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, keep only the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    # For debugging - create a color image to draw on
    debug_img = img.copy()
    
    # Try to find rectangular contours (like display panels)
    display_contours = []
    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Skip very small contours
        if area < 500:  # Adjust this threshold as needed
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Draw contour index for debugging
        cv2.putText(debug_img, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check if it looks like a seven-segment display panel
        # A display is typically wider than tall with a specific aspect ratio
        if 1.0 < aspect_ratio < 8.0 and w > 50 and h > 20:
            display_contours.append((x, y, w, h))
            # Draw rectangle for debugging
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Draw rejected contours in red
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Save debug image
    cv2.imwrite(os.path.join(output_dir, f"5_contours_{os.path.basename(image_path)}"), debug_img)
    
    # Sort display contours by y-position (top to bottom)
    display_contours = sorted(display_contours, key=lambda c: c[1])
    
    # Extract and process individual display regions
    weight_values = []
    
    for i, (x, y, w, h) in enumerate(display_contours):
        # Extract the display region (add a bit of padding)
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        display_region = img[y_start:y_end, x_start:x_end]
        display_region_gray = cv2.cvtColor(display_region, cv2.COLOR_BGR2GRAY)
        
        # Apply specific preprocessing for this region
        _, binary_region = cv2.threshold(display_region_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Save the extracted region for debugging
        cv2.imwrite(os.path.join(output_dir, f"6_region_{i}_{os.path.basename(image_path)}"), display_region)
        cv2.imwrite(os.path.join(output_dir, f"7_binary_{i}_{os.path.basename(image_path)}"), binary_region)
        
        # Try to recognize segments directly (advanced method)
        # This is a placeholder for specialized seven-segment recognition
        # For now, we'll use OCR but later we can implement a custom recognizer
        
        # Enhanced OCR approach: try multiple processing methods
        
        # 1. Try using the original region
        region_path = os.path.join(output_dir, f"ocr_region_{i}.jpg")
        cv2.imwrite(region_path, display_region)
        
        # 2. Try using the binary version
        binary_path = os.path.join(output_dir, f"ocr_binary_{i}.jpg")
        cv2.imwrite(binary_path, binary_region)
        
        # Use OCR on both versions
        ocr_results = []
        
        # Try regular image
        result_regular = ocr_space_file(region_path, language="eng", engine=2)
        if result_regular.get("IsErroredOnProcessing") == False:
            parsed = result_regular.get("ParsedResults")
            if parsed and parsed[0].get("ParsedText"):
                ocr_results.append(parsed[0]["ParsedText"].strip())
        
        # Try binary image
        result_binary = ocr_space_file(binary_path, language="eng", engine=2)
        if result_binary.get("IsErroredOnProcessing") == False:
            parsed = result_binary.get("ParsedResults")
            if parsed and parsed[0].get("ParsedText"):
                ocr_results.append(parsed[0]["ParsedText"].strip())
        
        # Process all OCR results
        for text in ocr_results:
            print(f"OCR on region {i}: '{text}'")
            
            # Clean the text to find numeric values
            # Remove all non-digit, non-decimal, non-unit characters
            cleaned_text = re.sub(r'[^0-9\.kgKG]', '', text)
            
            # Look for numbers and decimal points
            digits = re.findall(r'\d+\.?\d*', cleaned_text)
            if digits:
                # Check if kg/KG is in the string
                if 'kg' in text.lower() or 'kg' in cleaned_text.lower():
                    weight_values.append(f"{digits[0]} kg")
                else:
                    # If no unit is found, assume kg for simplicity in this context
                    weight_values.append(f"{digits[0]}")
        
        # Clean up temp files
        try:
            os.remove(region_path)
            os.remove(binary_path)
        except:
            pass
    
    # If no weights found with OCR, try more aggressive processing
    if not weight_values:
        # Try different thresholds and preprocessing
        # This is where custom seven-segment recognition would be implemented
        pass
    
    return weight_values

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
    
    # Try to detect seven-segment displays for weight values
    print("Attempting to detect seven-segment displays...")
    seven_segment_weights = detect_seven_segment_digits(image_path)
    if seven_segment_weights:
        print(f"Detected weights from display: {seven_segment_weights}")
        # Use the first detected weight
        if len(seven_segment_weights) > 0:
            package_weight = seven_segment_weights[0]
    else:
        print("No seven-segment display weights detected")
    
    # Iterate over the OCR data to extract word bounding boxes
    print("Processing regular OCR data...")
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
            
            # Extract detection information
            detected_text = word["WordText"]
            
            # Look for client ID (amk followed by 5 digits)
            client_id_matches = re.findall(r'\bamk\d{5}\b', detected_text.lower())
            if client_id_matches:
                client_id = client_id_matches[0]
                
            # Look for package weight (number followed by kg)
            if not package_weight:  # Only look if not already found in seven-segment display
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
    
    # Return the extracted information for potential further processing
    return {"client_id": client_id, "package_weight": package_weight}

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
                extracted_data = text_extraction(img_path, parsed_results)
                return extracted_data
            else:
                print("OCR completed but no text overlay found in response")
                print(f"Response structure: {list(parsed_results[0].keys())}")
        else:
            print("No text found in the image.")
    else:
        print("OCR Error:", result.get("ErrorMessage", "Unknown error occurred."))
        
    print("----------------------------")
    return None