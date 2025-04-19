import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def install_dependencies():
    """Install required packages if not already installed"""
    try:
        import paddleocr
    except ImportError:
        print("Installing PaddleOCR and dependencies...")
        os.system("pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install paddleocr")
        print("Installation complete")

def initialize_ocr(use_gpu=False):
    """
    Initialize PaddleOCR with Korean-English language support
    
    Args:
        use_gpu: Boolean indicating whether to use GPU acceleration
    
    Returns:
        PaddleOCR instance configured for Korean+English recognition
    """
    # PP-OCRv3 model is used by default in recent PaddleOCR versions
    ocr = PaddleOCR(
        use_angle_cls=True,    # Enable text direction classification
        lang="korean",         # Use Korean model which also supports English
        use_gpu=use_gpu,       # GPU acceleration if available
        show_log=False,        # Hide detailed logs
        det_db_thresh=0.3,     # Lower detection threshold for better text region detection
        det_db_box_thresh=0.5, # Box threshold for text detection
        rec_model_dir=None,    # Use default model directory
        det_model_dir=None,    # Use default detection model
        cls_model_dir=None     # Use default classification model
    )
    return ocr
def process_image(ocr_engine, image_path, is_file=True, visualize=False):
    """
    Process an image to detect and extract text from labels
    
    Args:
        ocr_engine: Initialized PaddleOCR instance
        image_path: Path to image file or numpy image array
        is_file: Boolean indicating if image_path is a file path (True) or numpy array (False)
        visualize: Whether to visualize results
    
    Returns:
        tuple: (original image, label image, OCR results, success flag)
    """
    try:
        if is_file:
            # Load image from file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            # Use provided numpy array
            img = image_path
            if not isinstance(img, np.ndarray):
                raise ValueError("Provided image_path is not a valid numpy array")
        
        # Resize image if it's too large
        max_dim = 1600
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Step 1: Detect the label in the image
        label_img, success = detect_label(img, visualize=visualize)
        
        if not success or label_img is None:
            print("No label detected in the image.")
            return img, None, None, False
        
        # Step 2: Preprocess the detected label
        processed_label = preprocess_label(label_img, visualize=visualize)
        
        # Convert BGR to RGB for OCR
        rgb_label = cv2.cvtColor(processed_label, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re
import imutils
from scipy.spatial import distance as dist

def install_dependencies():
    """Install required packages if not already installed"""
    try:
        import paddleocr
        import imutils
    except ImportError:
        print("Installing PaddleOCR, imutils and dependencies...")
        os.system("pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple")
        os.system("pip install paddleocr imutils")
        print("Installation complete")

def initialize_ocr(use_gpu=False):
    """
    Initialize PaddleOCR with Korean-English language support
    
    Args:
        use_gpu: Boolean indicating whether to use GPU acceleration
    
    Returns:
        PaddleOCR instance configured for Korean+English recognition
    """
    # PP-OCRv3 model is used by default in recent PaddleOCR versions
    ocr = PaddleOCR(
        use_angle_cls=True,    # Enable text direction classification
        lang="en",             # Use English model (change according to your labels)
        use_gpu=use_gpu,       # GPU acceleration if available
        show_log=False,        # Hide detailed logs
        det_db_thresh=0.3,     # Lower detection threshold for better text region detection
        det_db_box_thresh=0.5, # Box threshold for text detection
        rec_model_dir=None,    # Use default model directory
        det_model_dir=None,    # Use default detection model
        cls_model_dir=None     # Use default classification model
    )
    return ocr

def detect_label(image, visualize=False):
    """
    Detect shipping label in the image
    
    Args:
        image: Input image (OpenCV format, BGR)
        visualize: Whether to visualize intermediate steps
    
    Returns:
        Tuple: (cropped label image, success flag)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Perform morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    if visualize:
        cv2.imshow("Thresholded", thresh)
        cv2.imshow("Morphological Operations", morph)
        cv2.waitKey(0)
    
    # Find contours in the thresholded image
    contours = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # If no contours were found, try alternative approach
    if not contours:
        # Try Canny edge detection instead
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect edge fragments
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        if visualize:
            cv2.imshow("Canny Edges", edges)
            cv2.imshow("Dilated Edges", dilated)
            cv2.waitKey(0)
        
        # Find contours in the edge image
        contours = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
    
    # If still no contours, try direct rectangle detection with Hough Transform
    if not contours:
        # Try a more direct approach - look for a white label on the package
        # This assumes the label is generally brighter than the package
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Hough Lines to detect straight lines that could be the label edges
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # Create a blank mask
            mask = np.zeros_like(gray)
            
            # Draw the lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            
            # Dilate to connect nearby lines
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours in the mask
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
    
    # If still no contours, try a simpler approach - just look for bright regions
    if not contours:
        # Try simple thresholding to find the bright label
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphology
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if visualize:
            cv2.imshow("Binary Threshold", binary)
            cv2.waitKey(0)
        
        # Find contours
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
    
    # Last resort: if still no contours, return the full image
    if not contours:
        print("No label contours found, using full image")
        h, w = image.shape[:2]
        return image, True
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Take the largest contour that looks like a rectangle
    label_contour = None
    
    # First try to find a contour that's approximately a rectangle
    for c in contours[:5]:  # Check the 5 largest contours
        # Get the perimeter of the contour
        peri = cv2.arcLength(c, True)
        
        # Approximate the contour shape
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If it has 4-6 vertices, it might be a label
        if 4 <= len(approx) <= 6:
            label_contour = approx
            break
    
    # If no suitable rectangular contour is found, just use the largest contour
    if label_contour is None and contours:
        # Get the largest contour
        largest_contour = contours[0]
        
        # Get a bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create a rectangular contour
        label_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    
    # If we still don't have a contour, use the full image
    if label_contour is None:
        print("No suitable label contour found, using full image")
        h, w = image.shape[:2]
        return image, True
    
    # Draw the detected label contour if visualization is requested
    if visualize:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [label_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Detected Label", debug_img)
        cv2.waitKey(0)
    
    # Get a bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(label_contour)
    
    # Crop the image to the bounding rectangle
    cropped = image[y:y+h, x:x+w]
    
    return cropped, True

def preprocess_label(label_image, visualize=False):
    """
    Preprocess the detected label for better OCR
    
    Args:
        label_image: The cropped label image
        visualize: Whether to visualize preprocessing steps
    
    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Binarization using Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Reduce noise with a median filter
    denoised = cv2.medianBlur(binary, 3)
    
    if visualize:
        plt.figure(figsize=(12, 10))
        plt.subplot(221), plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)), plt.title('Original Label')
        plt.subplot(222), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
        plt.subplot(223), plt.imshow(enhanced, cmap='gray'), plt.title('CLAHE Enhanced')
        plt.subplot(224), plt.imshow(denoised, cmap='gray'), plt.title('Denoised Binary')
        plt.tight_layout()
        plt.show()
    
    # Convert back to BGR for PaddleOCR compatibility
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return processed

def process_image(ocr_engine, image_path, is_file=True, visualize=False):
    """
    Process an image to detect and extract text from labels
    
    Args:
        ocr_engine: Initialized PaddleOCR instance
        image_path: Path to image file or numpy image array
        is_file: Boolean indicating if image_path is a file path (True) or numpy array (False)
        visualize: Whether to visualize results
    
    Returns:
        tuple: (original image, label image, OCR results, success flag)
    """
    if is_file:
        # Load image from file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
    else:
        # Use provided numpy array
        img = image_path
        if not isinstance(img, np.ndarray):
            raise ValueError("Provided image_path is not a valid numpy array")
    
    # Resize image if it's too large
    max_dim = 1600
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Step 1: Detect the label in the image
    label_img, success = detect_label(img, visualize=visualize)
    
    if not success:
        print("No label detected in the image.")
        return img, None, None, False
    
    # Step 2: Preprocess the detected label
    processed_label = preprocess_label(label_img, visualize=visualize)
    
    # Convert BGR to RGB for OCR
    rgb_label = cv2.cvtColor(processed_label, cv2.COLOR_BGR2RGB)
    
    # Step 3: Run OCR on the preprocessed label
    result = ocr_engine.ocr(rgb_label, cls=True)
    
    return img, label_img, result, True

def extract_client_ids(result, prefix="AMK", min_confidence=0.6):
    """
    Extract client IDs starting with the specified prefix
    
    Args:
        result: OCR results from PaddleOCR
        prefix: The prefix to filter for (default: "AMK")
        min_confidence: Minimum confidence threshold
    
    Returns:
        list: List of (client_id, confidence, bounding box) tuples
    """
    client_ids = []
    
    # Handle case where result is None
    if result is None:
        return client_ids
        
    # Handle PaddleOCR results structure
    if isinstance(result, list) and len(result) > 0:
        # Check if there are any text detections
        if len(result[0]) > 0:
            for line in result[0]:
                # Check if the line has the expected structure
                if len(line) >= 2 and isinstance(line[0], list) and isinstance(line[1], list) and len(line[1]) >= 2:
                    bbox = line[0]  # Bounding box
                    text = line[1][0]  # Detected text
                    confidence = line[1][1]  # Confidence score
                    
                    # Skip low confidence results
                    if confidence < min_confidence:
                        continue
                        
                    # Check if text contains the prefix (case insensitive)
                    if prefix.upper() in text.upper():
                        # Extract the full client ID using regex
                        # Pattern looks for prefix followed by alphanumeric characters
                        pattern = f"{prefix}[A-Z0-9]+"
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        
                        if matches:
                            for match in matches:
                                client_ids.append((match, confidence, bbox))
    
    return client_ids

def visualize_results(image, label_image, ocr_result, client_ids, output_path=None):
    """
    Visualize detection and OCR results
    
    Args:
        image: Original image
        label_image: Detected and cropped label image
        ocr_result: OCR results
        client_ids: Extracted client IDs
        output_path: Path to save visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Display detected label
    if label_image is not None:
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Label')
        plt.axis('off')
    else:
        plt.subplot(222)
        plt.text(0.5, 0.5, "No label detected", 
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    # Display OCR results on label
    if label_image is not None and ocr_result is not None and isinstance(ocr_result, list) and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
        plt.subplot(223)
        
        try:
            # Convert to PIL Image for drawing
            img_pil = Image.fromarray(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
            
            # Prepare OCR visualization data
            boxes = [line[0] for line in ocr_result[0]]
            txts = [line[1][0] for line in ocr_result[0]]
            scores = [line[1][1] for line in ocr_result[0]]
            
            # Draw OCR results on image
            try:
                # Try with default font
                drawn_img = draw_ocr(img_pil, boxes, txts, scores, font_path=None)
                plt.imshow(drawn_img)
            except Exception as e:
                # If default font fails, use a simpler approach
                print(f"Warning: Could not draw OCR results with default font: {str(e)}")
                # Create a copy of the image for drawing
                vis_img = label_image.copy()
                
                # Draw each bounding box and text
                for box, txt, score in zip(boxes, txts, scores):
                    box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(vis_img, [box], True, (0, 255, 0), 2)
                    
                    # Get centroid of the box for text placement
                    box_flat = box.reshape(-1, 2)
                    centroid_x = int(np.mean(box_flat[:, 0]))
                    centroid_y = int(np.mean(box_flat[:, 1]))
                    
                    # Put text with score
                    cv2.putText(vis_img, f"{txt[:10]}... ({score:.2f})", 
                               (centroid_x, centroid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                
            plt.title('OCR Results')
        except Exception as e:
            # If any error occurs in visualization, show the label image
            print(f"Error visualizing OCR results: {str(e)}")
            plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
            plt.title('OCR Results (visualization failed)')
    else:
        plt.subplot(223)
        if label_image is not None:
            plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
        plt.title('No OCR Results')
    
    plt.axis('off')
    
    # Display client IDs
    plt.subplot(224)
    if label_image is not None and client_ids:
        try:
            # Create a copy of label image for highlighting client IDs
            highlight_img = label_image.copy()
            
            for client_id, confidence, bbox in client_ids:
                # Convert bbox points to integer
                bbox = np.array(bbox).astype(np.int32)
                
                # Draw bounding box with thicker line for client IDs
                cv2.polylines(highlight_img, [bbox.reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                
                # Calculate center for text placement
                center_x = np.mean(bbox[:, 0])
                center_y = np.mean(bbox[:, 1])
                
                # Add text with ID and confidence
                cv2.putText(highlight_img, f"{client_id} ({confidence:.2f})", 
                           (int(center_x), int(center_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            plt.imshow(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
            plt.title('Detected Client IDs')
        except Exception as e:
            print(f"Error highlighting client IDs: {str(e)}")
            plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
            plt.title('Client IDs (visualization failed)')
    else:
        plt.text(0.5, 0.5, "No client IDs detected", 
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {output_path}")
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Error showing visualization: {str(e)}")
        plt.close()

def process_from_camera(ocr_engine=None, camera_id=0, use_gpu=False):
    """
    Process frames from camera and detect client IDs in real-time
    
    Args:
        ocr_engine: PaddleOCR instance (created if None)
        camera_id: Camera device ID
        use_gpu: Whether to use GPU for OCR
    """
    # Initialize OCR if not provided
    if ocr_engine is None:
        ocr_engine = initialize_ocr(use_gpu)
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera active. Press 'q' to quit, 'c' to capture and process a frame.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Display live feed
        cv2.imshow("Delivery Box Scanner (Press 'c' to capture, 'q' to quit)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit on 'q'
            break
        elif key == ord('c'):
            # Process current frame on 'c'
            print("Processing frame...")
            try:
                # Make a copy of the frame to prevent modification of the displayed frame
                frame_copy = frame.copy()
                
                # Process the current frame
                _, label_img, result, success = process_image(ocr_engine, frame_copy, is_file=False, visualize=False)
                
                if success:
                    # Extract client IDs
                    client_ids = extract_client_ids(result)
                    
                    if client_ids:
                        print("\nDetected Client IDs:")
                        for idx, (client_id, confidence, _) in enumerate(client_ids):
                            print(f"[{idx+1}] ID: {client_id} (Confidence: {confidence:.4f})")
                    else:
                        print("No client IDs detected in this frame.")
                    
                    # Visualize results
                    visualize_results(frame_copy, label_img, result, client_ids)
                else:
                    print("No label detected in this frame.")
            
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def batch_process_directory(directory_path, output_dir=None, use_gpu=False):
    """
    Batch process all images in a directory
    
    Args:
        directory_path: Path to directory containing images
        output_dir: Directory to save results (created if doesn't exist)
        use_gpu: Whether to use GPU for OCR
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize OCR
    ocr_engine = initialize_ocr(use_gpu)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images. Starting batch processing...")
    
    # Process each image
    results = []
    for i, filename in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        image_path = os.path.join(directory_path, filename)
        try:
            # Process image
            img, label_img, ocr_result, success = process_image(ocr_engine, image_path)
            
            if success:
                # Extract client IDs
                client_ids = extract_client_ids(ocr_result)
                
                # Save visualization if output directory specified
                if output_dir is not None:
                    output_path = os.path.join(output_dir, f"result_{os.path.splitext(filename)[0]}.png")
                    visualize_results(img, label_img, ocr_result, client_ids, output_path)
                
                # Add to results
                results.append({
                    'filename': filename,
                    'success': True,
                    'client_ids': [(cid, conf) for cid, conf, _ in client_ids] if client_ids else []
                })
                
                # Print client IDs
                if client_ids:
                    print(f"  Detected {len(client_ids)} client ID(s):")
                    for idx, (client_id, confidence, _) in enumerate(client_ids):
                        print(f"    [{idx+1}] ID: {client_id} (Confidence: {confidence:.4f})")
                else:
                    print("  No client IDs detected.")
            else:
                print("  No label detected in the image.")
                results.append({
                    'filename': filename,
                    'success': False,
                    'client_ids': []
                })
        
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e),
                'client_ids': []
            })
    
    # Generate summary
    print("\nProcessing Summary:")
    print(f"Total images: {len(image_files)}")
    success_count = sum(1 for r in results if r['success'])
    print(f"Successful detections: {success_count} ({success_count/len(image_files)*100:.1f}% if len(image_files) > 0 else 0%)")
    
    detected_ids_count = sum(len(r['client_ids']) for r in results)
    print(f"Total client IDs detected: {detected_ids_count}")
    
    # Return results for potential further analysis
    return results

def main():
    """Main function to demonstrate capabilities"""
    # Install dependencies if needed
    install_dependencies()
    
    print("Delivery Box Label Detection and OCR System")
    print("===========================================")
    print("1. Process a single image")
    print("2. Process images from camera")
    print("3. Batch process directory")
    print("q. Quit")
    
    choice = input("\nEnter your choice: ")
    
    if choice == '1':
        image_path = input("Enter image path: ")
        use_gpu = input("Use GPU? (y/n): ").lower() == 'y'
        
        # Initialize OCR
        ocr_engine = initialize_ocr(use_gpu)
        
        # Process image
        try:
            img, label_img, ocr_result, success = process_image(ocr_engine, image_path, visualize=True)
            
            if success:
                # Extract client IDs
                client_ids = extract_client_ids(ocr_result)
                
                # Visualize results
                visualize_results(img, label_img, ocr_result, client_ids)
                
                # Print client IDs
                if client_ids:
                    print("\nDetected Client IDs:")
                    for idx, (client_id, confidence, _) in enumerate(client_ids):
                        print(f"[{idx+1}] ID: {client_id} (Confidence: {confidence:.4f})")
                else:
                    print("No client IDs detected.")
            else:
                print("No label detected in the image.")
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif choice == '2':
        use_gpu = input("Use GPU? (y/n): ").lower() == 'y'
        try:
            camera_id = int(input("Enter camera ID (usually 0 for built-in): "))
            process_from_camera(camera_id=camera_id, use_gpu=use_gpu)
        except ValueError:
            print("Invalid camera ID. Using default (0).")
            process_from_camera(use_gpu=use_gpu)
    
    elif choice == '3':
        directory = input("Enter directory path containing images: ")
        output_dir = input("Enter output directory path (or press Enter to skip saving): ")
        if not output_dir:
            output_dir = None
        use_gpu = input("Use GPU? (y/n): ").lower() == 'y'
        
        batch_process_directory(directory, output_dir, use_gpu)
    
    elif choice.lower() == 'q':
        print("Exiting.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()