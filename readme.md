# Optical Character Recognition (OCR) on Postal Images

## 👥 Team Members
- **Chaw Thiri San** — Team Leader & ML Engineer  
- Shukurullo Meliboev  
- Ilhomov Mansur Ilhom Ugli  
- Javokhir Sirojiddinov  
- Irmuun  
- Adkham Bakhromov  
- Mukhammadov Muminjon  
- Adilet Azhikeev  

---

## 📌 Project Overview
This repository contains code for multilingual Optical Character Recognition (OCR) on postal images. The system extracts three types of information from each image:

1. **Client ID**  
2. **Barcode**  
3. **Package Weight**

It supports both computer and Raspberry Pi environments with optimized models for each platform.

---

## 📁 Repository Structure
- `computer_version/` – Scripts for running on desktop computers (higher processing power).
- `raspberry_version/` – Optimized code for running on Raspberry Pi.
- `seven_segment_detection/` – YOLOv7-based model for detecting digits on 7-segment displays.
- `archived_version/` – Older versions of models and experiments.

---

## 📊 Sample Results
![image](https://github.com/user-attachments/assets/18f2d4d0-1974-45f5-93ed-e834227ff99d)
![image](https://github.com/user-attachments/assets/5a352e44-8623-4468-ae74-55b8cc30babd)
![image](https://github.com/user-attachments/assets/d1a46462-e04e-4917-87aa-f4b955cc40b0)

---

## 🛠 Tech Stack
**Libraries:**  
- `matplotlib`, `numpy`, `opencv-python`, `paddleocr`, `Pillow`, `requests`, `python-dotenv`, `pyzbar`

**Third-Party Tools:**  
- **Roboflow** – For training the 7-segment digit detection model  
- **OCR.Space API** – For lightweight OCR on Raspberry Pi

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/chaw-thiri/Digital_Signal_Processing.git
   ```
### 2. Navigate to the project directory:
   ```bash
   cd Digital_Signal_Processing
   ```
### 3. Install dependencies (depending on the device you are running ):
   * 📍We highly recommend you to use a virtual environment. Skip this part if you are downloading directly to your global system.
     ```
     python -m venv venv
      venv\\Scripts\\activate   # Windows
      source venv/bin/activate  # macOS/Linux

     ```
   * Each folder has its own requirements.txt. Choose the appropriate one depending on your device (PC or Raspberry Pi):
     ```bash
   
         pip install -r computer_version/requirements.txt
         # or
         pip install -r raspberry_version/requirements.txt

      ```
### 4. Run the scripts or Jupyter notebooks to explore the implementations.

# ✨Key Features
## 🎀Paddle OCR model
* Designed for bulk processing on systems with strong processing power.
* Includes parallel processing for improved performance.
* Currently supports Korean and English.
* Additional languages can be integrated upon request.
## 🍓 Raspberry Pi model
* Being lightweight is a key of a successful model on raspberry pi and this is what we are proud of our model. Without any additional configuration the model can detect 27 languages.
## 7️⃣ 7 segment digits extractor model 
* Utilizes YOLO11 for accurate detection of digital weight displays.
* Please go to this [readme](https://github.com/chaw-thiri/Digital_Signal_Processing/blob/main/seven_segment_detection/readme_7segment.md) link for more detail.

## Technical Details
Details available in this [powerpoint](https://www.canva.com/design/DAGmIHlJc2k/0U2dUyBtWBXQkvOeobpH-A/edit?utm_content=DAGmIHlJc2k&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## Contribution Guidelines
- Use feature branches for new implementations.
- Commit messages should be clear and descriptive.
- Pull requests should be reviewed by at least one team member before merging.

## Contact
For any questions or collaborations, please reach out to any of the team members.

---
_This repository is maintained as part of our Digital Signal Processing coursework._

