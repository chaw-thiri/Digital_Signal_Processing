# from img_processing import image_cropper
from ocr import run_single_img_ocr
from barcode import detect_barcodes
import numpy as np
import os 

# TODO : Crop the sides of the image
# TODO : Add barcode scanner
# TODO : ADD bulk processing
# TODO : add data base
# TODO : add webcam processing
# TODO : add autorotation
# TODO : add parallel processing


# -------------------------------------------------------- Single image test ----------------
#img_path = r"Postal-DB-dataset\content\Postal-DB\cropped_dataset\testimg.jpg"
#run_single_img_ocr(img_path=img_path)

# --------------------------------------------------------- Folder of images test ------------
img_folder = "/Users/shukurullomeliboyev2004/Desktop/Digital_Signal_Processing/cropped_dataset"
# Loop through all files in the folder
for filename in os.listdir(img_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(img_folder, filename)
        run_single_img_ocr(img_path=img_path)
        detect_barcodes(image_path=img_path)
