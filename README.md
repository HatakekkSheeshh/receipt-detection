# Receipt Detection and OCR

## Overview

This project provides real-time receipt detection from a webcam using YOLOv8, with integrated OCR (Optical Character Recognition) powered by EasyOCR. Detected receipts are highlighted with bounding boxes, and text is extracted from each detected region.

---

## Features
- Real-time object detection using YOLOv8
- Automatic text extraction from detected receipts using EasyOCR
- Modular code structure for easy extension and maintenance

---

## Installation

1. **Clone the repository and navigate to the project folder.**
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Place your trained YOLO model (`best.pt`) in the `data/` directory.**
2. **Run the main program:**
   ```bash
   python main.py
   ```
3. **The webcam will open.**
   - Receipts will be detected and shown with bounding boxes.
   - Detected text will be printed in the console.
   - Press `ESC` to exit.

---

## Configuration & Notes
- To use a different camera, change the index in `cv2.VideoCapture(0)` in `yolo/detection.py`.
- To use a different YOLO model, update the model path in `yolo/detection.py`.
- If you encounter missing library errors, ensure all packages in `requirements.txt` are installed.
- For best OCR results, ensure good lighting and camera focus.

---

## Project Structure
- `yolo/` — YOLO detection logic and utilities
- `ocr/` — OCR processing utilities (EasyOCR integration)
- `data/` — Place your YOLO model weights here
- `main.py` — Entry point for running the application

---

## Contact
For questions or contributions, please open an issue or submit a pull request.

