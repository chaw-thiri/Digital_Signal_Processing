import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
# -------------------------------Prepare depedencies & OCR machine -----------------------------
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
# ------------------------------ Image Preprocessing ------------------------------
def process_image(ocr_engine, image_path, is_file=True):
    """
    Process an image for OCR
    
    Args:
        ocr_engine: Initialized PaddleOCR instance
        image_path: Path to image file or numpy image array
        is_file: Boolean indicating if image_path is a file path (True) or numpy array (False)
    
    Returns:
        tuple: (original image, OCR results, image dimensions)
    """
    if is_file:
        # Load image from file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Use provided numpy array
        img = image_path
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img
    
    # Get image dimensions
    h, w = img_rgb.shape[:2]
    
    # Run OCR
    result = ocr_engine.ocr(img_rgb, cls=True)
    
    return img_rgb, result, (w, h)
# ------------------------------------ Vistualization Output  -----------------------
def visualize_results(image, result, output_path=None, font_path="./fonts/korean.ttf"):
    """
    Visualize OCR results on the image
    
    Args:
        image: Original image (RGB format)
        result: OCR results from PaddleOCR
        output_path: Path to save visualization (if None, only displays)
        font_path: Path to font file for text rendering
    """
    # Convert to PIL Image
    img_pil = Image.fromarray(image)
    
    # Prepare visualization data
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    
    # Check if font exists, if not use default
    if not os.path.exists(font_path):
        print(f"Warning: Font file {font_path} not found. Using default font.")
        # Try to download a suitable font for Korean
        font_dir = "./fonts"
        os.makedirs(font_dir, exist_ok=True)
        
        try:
            import requests
            font_url = "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.6/doc/fonts/korean.ttf"
            r = requests.get(font_url)
            with open(os.path.join(font_dir, "korean.ttf"), "wb") as f:
                f.write(r.content)
            font_path = os.path.join(font_dir, "korean.ttf")
            print(f"Downloaded Korean font to {font_path}")
        except:
            print("Failed to download Korean font. Using system default.")
            font_path = None
    
    # Draw annotations on image
    draw_img = draw_ocr(img_pil, boxes, txts, scores, font_path=font_path)
    
    # Convert back to numpy for display
    draw_img = np.array(draw_img)
    
    # Display the result
    plt.figure(figsize=(15, 15))
    plt.imshow(draw_img)
    plt.axis('off')
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def extract_text(result):
    """
    Extract plain text from OCR results
    
    Args:
        result: OCR results from PaddleOCR
    
    Returns:
        list: List of (text, confidence) tuples
    """
    texts = []
    if result and len(result) > 0:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            texts.append((text, confidence))
    
    return texts
# ------------------------------------ OCR on Images  --------------------------

def run_imageOCR(image_path, use_gpu=False, visualize=True, output_path=None, ocr_engine=None):
    """
    Main function to perform OCR on an image
    
    Args:
        image_path: Path to the image file
        use_gpu: Boolean indicating whether to use GPU
        visualize: Boolean indicating whether to visualize results
        output_path: Path to save visualization (if None, only displays)
        ocr_engine: Optional pre-initialized PaddleOCR instance
    
    Returns:
        list: Extracted text with confidence scores
    """
    
    # Install dependencies if needed
    install_dependencies()
    
    # Initialize OCR engine if not provided
    if ocr_engine is None:
        print("Initializing PaddleOCR with PP-OCRv3 for Korean+English...")
        ocr_engine = initialize_ocr(use_gpu)
    
    # Process image
    print(f"Processing image: {image_path}")
    try:
        image, result, dimensions = process_image(ocr_engine, image_path)
        
        # Extract and print text
        texts = extract_text(result)
        print("\nExtracted Text:")
        for idx, (text, confidence) in enumerate(texts):
            print(f"[{idx+1}] Text: {text} (Confidence: {confidence:.4f})")
        
        # Visualize results if requested
        if visualize:
            visualize_results(image, result, output_path)
        
        return texts
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []

def run_image_ocr_batch(test_images_dir, output_dir, use_gpu=False):
    """
    Run OCR on all images in the specified directory.
    
    Args:
        test_images_dir: Path to folder containing test images.
        output_dir: Path to save visualization outputs.
        use_gpu: Use GPU acceleration if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(test_images_dir, "*.png")) +
        glob.glob(os.path.join(test_images_dir, "*.jpg")) +
        glob.glob(os.path.join(test_images_dir, "*.jpeg"))
    )

    print(f"Found {len(image_paths)} images to process.")

    # Initialize OCR once
    print("Initializing PaddleOCR once for all images...")
    ocr_engine = initialize_ocr(use_gpu=use_gpu)

    for idx, image_path in enumerate(image_paths, start=1):
        output_path = os.path.join(output_dir, f"img{idx}.png")
        print(f"\n--- Processing {os.path.basename(image_path)} ---")
        run_imageOCR(image_path, use_gpu=use_gpu, visualize=True, output_path=output_path, ocr_engine=ocr_engine)

# ------------------------------------ OCR on Webcam Video ----------------------
def run_webcam_ocr(use_gpu=False):
    """
    Run OCR using the webcam feed. Press 'c' to capture a frame and perform OCR.
    Press 'q' to quit the webcam window.
    
    Args:
        use_gpu: Boolean indicating whether to use GPU acceleration
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize OCR engine
    ocr_engine = initialize_ocr(use_gpu)

    print("Webcam OCR started. Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow('Press "c" to capture for OCR | "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Perform OCR on captured frame
            try:
                image, result, _ = process_image(ocr_engine, frame, is_file=False)
                texts = extract_text(result)

                print("\nExtracted Text:")
                for idx, (text, confidence) in enumerate(texts):
                    print(f"[{idx+1}] Text: {text} (Confidence: {confidence:.4f})")

                # Draw bounding boxes
                for line in result[0]:
                    box = np.array(line[0], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [box], True, (0, 255, 0), 2)

                cv2.imshow('OCR Result', frame)

            except Exception as e:
                print(f"OCR error: {str(e)}")

        elif key == ord('q'):
            print("Quitting webcam OCR.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # IMG OCR
    test_images_dir = "/content/Digital_Signal_Processing/test_images"
    output_dir = "/content/Digital_Signal_Processing/detected_images"
    single_img = "/content/Digital_Signal_Processing/test_images/test1.png"
    
    #               running OCR on single img
    # run_imageOCR(single_img, use_gpu= False, visualize= True, output_path="OCR_image")
    #               running OCR on multiple imgs
    #run_image_ocr_batch(test_images_dir, output_dir, use_gpu=False)

    # WEBCAM OCR
    # run_webcam_ocr(use_gpu=False)
   
    