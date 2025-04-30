import os
from ocr import run_single_img_ocr
import cv2

def process_image_folder(folder_path):
    """Process all images in the dataset folder"""
    print(f"[*] Processing images in: {folder_path}")
    processed = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            print(f"\n[*] Processing: {filename}")
            run_single_img_ocr(img_path)
            processed += 1
    print(f"\n[✔] Processed {processed} images")

def capture_from_webcam(save_path="capture.jpg"):
    """Capture image from Raspberry Pi camera or USB webcam"""
    cap = cv2.VideoCapture(0)
    print("[*] Capturing image from webcam...")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
        print(f"[✔] Saved capture to {save_path}")
        run_single_img_ocr(save_path)
    else:
        print("[❌] Failed to capture image.")
    cap.release()

if __name__ == "__main__":
    # Set default dataset path relative to project root
    default_dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    
    print("Choose mode:")
    print("1. Process images from dataset folder")
    print("2. Capture from webcam")
    mode = input("Enter choice (1/2): ").strip()

    if mode == "1":
        if os.path.isdir(default_dataset):
            process_image_folder(default_dataset)
        else:
            print(f"[!] Dataset folder not found at: {default_dataset}")
            custom_path = input("Enter custom dataset path: ").strip()
            if os.path.isdir(custom_path):
                process_image_folder(custom_path)
            else:
                print("[!] Invalid folder path.")
    elif mode == "2":
        capture_from_webcam()
    else:
        print("[!] Invalid option.") 