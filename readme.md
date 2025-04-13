# Digital Signal Processing (DSP) Project

## Team Members
* **Chaw Thiri San** (Team Leader) - ID: 12225272
* **Shukurullo Meliboev** - ID: 12225261
* **Ilhomov Mansur Ilhom Ugli** - ID: 12225247
* **Irmuun Munkhtulga** - ID: 12220272

## Project Overview
This repository contains our coursework and project files for the Digital Signal Processing (DSP) course. Our work includes signal processing techniques, implementation of various DSP algorithms, and practical applications using Python and MATLAB.

## Prerequisites

### System Requirements
- Python 3.7 or higher
- Tesseract OCR
- Korean language support for Tesseract

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/chaw-thiri/Digital_Signal_Processing.git
   cd Digital_Signal_Processing
   ```

2. Install system dependencies:
   ```bash
   # For macOS testing
   brew install tesseract
   brew install tesseract-lang

   # For Ubuntu/Debian in RaspberryPi kit
   sudo apt-get install tesseract-ocr
   sudo apt-get install tesseract-ocr-kor
   ```

3. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
Digital_Signal_Processing/
├── test_images/          # Test images for OCR
├── paddleOCR/            # PaddleOCR implementation
│   ├── detected_images/  # Output directory for processed images
│   ├── fonts/           # Font files for visualization
│   └── paddleOCR.py     # Main PaddleOCR implementation
└── pytesseract/         # Tesseract implementation
    └── main.py          # Main Tesseract implementation
```

## How to Run the Code

### Testing Pytesseract Implementation
```bash
python pytesseract/main.py
```

### Testing PaddleOCR Implementation
```bash
python paddleOCR/paddleOCR.py
```

## Features
- OCR implementation using both PaddleOCR and Tesseract
- Support for Korean and English text recognition
- Image preprocessing and visualization
- Batch processing of multiple images
- Webcam support for real-time OCR

## Technologies Used
* **Programming Languages:** Python
* **Libraries & Tools:** 
  - PaddlePaddle
  - PaddleOCR
  - Tesseract OCR
  - OpenCV
  - NumPy
  - Matplotlib
  - Pillow

## Contribution Guidelines
- Use feature branches for new implementations.
- Commit messages should be clear and descriptive.
- Pull requests should be reviewed by at least one team member before merging.

## Contact
For any questions or collaborations, please reach out to any of the team members.

---
_This repository is maintained as part of our Digital Signal Processing coursework._

