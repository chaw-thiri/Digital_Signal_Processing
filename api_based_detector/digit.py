import numpy as np
import cv2
import os

def detect_box_sequences(image_path):
    if isinstance(image_path, str):  # If the input is a file path
        assert os.path.isfile(image_path), "File [%s] doesn't exist!" % image_path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path, np.ndarray):  # If the input is already a numpy array (image)
        image = image_path
    else:
        raise ValueError("Unsupported input type: expected file path or numpy array.")

    # Ensure the image is in uint8 format (necessary for contour finding)
    if image.dtype != 'uint8':
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype('uint8')

    # Your processing code, e.g., find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours