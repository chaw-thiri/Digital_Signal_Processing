
import cv2

def image_cropper(img_path):
    # crop 1/3 from both left and right
    # THIS FUNCTION IS ONLY USEFUL FOR THE MID-TERM DATASET 
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    crop_margin = w // 3
    cropped_img = img[:, crop_margin: w- crop_margin]
    return cropped_img