# 7-Segment Display Digit Detection (YOLOv11)

## Getting Started

To download the necessary base files, run:

```bash
pip install inference-cli && inference server start
```
## Model Overview
* This project utilizes a **YOLOv11 model** specifically trained to detect digits on 7-segment displays.
* Training Dataset: 949 public images + 21 private images

## Performance Metrics:
* mAP@50: 91.4%
* Precision: 89.6%
* Recall: 92.0%

The model accurately detects: Digits 0–9 and the decimal point (".")

**Note: Labels for the private dataset were manually annotated using MakeSense.ai in YOLO format.**

## Training Graph
 ![image](https://github.com/user-attachments/assets/0966e9c9-c67c-4c44-8203-63e9c3b9a631)
 

## Pre-processing
Each image was processed with:
* Auto-orientation (EXIF metadata stripped)
* Resize to 640×640 (stretched)
* Auto-contrast using adaptive equalization

### Augmentation
Each original image was augmented into three versions using: Salt-and-pepper noise on 0.1% of pixels

## Dataset Split & Final Count
The dataset was split into: Training: 60%, Validation: 20% & Testing: 20%.     
After augmentation, the final dataset includes 2374 images: 2106 training images, 168 validation images, 100 testing images

### Download Dataset
📦 Download the augmented dataset on Roboflow : [LINK](https://app.roboflow.com/ds/kaAzg6kmsp?key=T7rW9qSNWS) 

This repository will be made private after grading to protect the privacy of private data contributors.

## Prediction result : 
Sample Output 1    
![image](https://github.com/user-attachments/assets/5a352e44-8623-4468-ae74-55b8cc30babd)    
Sample Output 2      
![image](https://github.com/user-attachments/assets/b8b27b6d-9178-42fa-9f48-04841a518a83)     
Sample Output 3      
![image](https://github.com/user-attachments/assets/46c26d7a-59fd-4e7d-9f40-df85b2e592ca)   

