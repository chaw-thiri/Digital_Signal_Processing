from ocr import run_single_img_ocr
import os
import traceback  # For better error reporting

# TODO : Crop the sides of the image
# TODO : Add barcode scanner
# TODO : ADD bulk processing
# TODO : add data base
# TODO : add webcam processing
# TODO : add autorotation
# TODO : add parallel processing

def process_images():
    """Process all images in the specified folder"""
    # --------------------------------------------------------- Single image test ------------
    # Uncomment to process a single image
    # img_path = "/Users/shukurullomeliboyev2004/Desktop/Digital_Signal_Processing/20250403_114728.jpg"
    # run_single_img_ocr(img_path=img_path)

    # --------------------------------------------------------- Folder of images test ------------
    img_folder = "/Users/shukurullomeliboyev2004/Desktop/Digital_Signal_Processing/cropped_dataset"
    
    # Check if the folder exists
    if not os.path.exists(img_folder):
        print(f"Error: Folder not found: {img_folder}")
        return
        
    # Count valid image files
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in folder: {img_folder}")
        return
        
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory for processed images
    output_dir = os.path.join(img_folder, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Loop through all files in the folder
    for i, filename in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
        img_path = os.path.join(img_folder, filename)
        
        try:
            run_single_img_ocr(img_path=img_path)
        except Exception as e:
            print(f"Error processing {filename}:")
            traceback.print_exc()

if __name__ == "__main__":
    process_images()