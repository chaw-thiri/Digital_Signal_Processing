import pytesseract
import cv2

image = cv2.imread("test1.png")
text = pytesseract.image_to_string(image, lang="kor")
print(text)