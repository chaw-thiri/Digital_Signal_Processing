
from dotenv import load_dotenv
from img_processing import image_cropper
import os
import re
import requests
from PIL import Image, ImageDraw, ImageFont
import json
from os import path
load_dotenv()
API_KEY = os.getenv("OCR_SPACE_API_KEY")


# Parameters for text overlay 
unicode_font_name = "./Arial Unicode.ttf"
TINT_COLOR = (255, 255, 0) 
TRANSPARENCY = .70  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)
unicode_font_name  = r"mid-term project\Digital_Signal_Processing\fonts\Arial Unicode.ttf"


def text_extraction(image_path, ocr_data):
    # Read the image
    
    img = Image.open(image_path)
    img = img.convert("RGBA")

    overlay = Image.new('RGBA', img.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(overlay) 


    
    # Iterate over the OCR data to extract word bounding boxes
    for line in ocr_data[0]['TextOverlay']['Lines']:
        img_id = 1
        word_max_height = line['MaxHeight']
        top_min_edge_to_phrase_distance = line["MinTop"]
        for word in line['Words']:
            x1 = (word["Left"], word["Top"])
            x2 = (x1[0] + word["Width"], x1[1] + word["Height"])

            # Adjust font size according to the rectangle height
            font_size = abs(x1[1] - x2[1])
            font = ImageFont.truetype(unicode_font_name, int(font_size))

            draw.rectangle((x1, x2), fill=TINT_COLOR+(OPACITY,))

            detected_text = word["WordText"]
           

            draw.text(x1, detected_text, fill=(255, 0, 0, 255), font=font)


            # EXTRACT CLIENT ID from all the detections 
            clientId = re.findall(r'\bamk\d{5}\b', detected_text.lower())
            package_weight = re.findall(r'\b\d+(?:\.\d+)?\s*kg\b', detected_text.lower())

            if clientId and package_weight:
                print(f"ID : {clientId[0]}, Package weight :{package_weight[0]}")
            elif clientId:
                print(f"ID : {clientId[0]}, No weight detected ")
            elif package_weight:
                print(f"Package weight: {package_weight[0]}, no ID present ")
            else: 
                print("Detection fails")

            
        img_id += 1
    
    img = Image.alpha_composite(img, overlay)

    output_file_name = f"img_{img_id}_overlay.png"
    img.save(output_file_name)
    # img.show()

def ocr_space_file(filename, api_key= API_KEY, overlay=True, language= "auto", engine=2):

    """ OCR.space API request with local file.
        :param filename: Path to the image file.
        :param api_key: Your OCR.space API key (default is 'helloworld' - demo key).
        :param overlay: Is OCR overlay required in the response.
        :param language: OCR language(s), separated by commas.
        :param engine: 1 or 2 (Engine 2 supports language auto-detection).
        :return: Result in JSON format.
    """
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
        'OCREngine': engine

    }
    with open(filename, 'rb') as f:
        r = requests.post(
        'https://api.ocr.space/parse/image',
        files={'file': f},
        data=payload,
         )


    return r.json()

def run_single_img_ocr(img_path):
    # Example usage
    #img_path = image_cropper(img_path)
    result = ocr_space_file(img_path)

    # Print the detected text
    # Check if there is no error
    if result.get("IsErroredOnProcessing") == False:
        parsed_results = result.get("ParsedResults")
        
        if parsed_results:
            # Load the image using OpenCV
            
            text_extraction(img_path, parsed_results)
        else:
            print("No text found.")
    else:
        print("Error:", result.get("ErrorMessage", "Unknown error occurred."))

