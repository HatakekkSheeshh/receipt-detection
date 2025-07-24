# receipt-detection

## Description 

Detect receipts in real-time from webcam using YOLOv8.  

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Usage 

1. Place your trained YOLO model (`best.pt`) in the `data/` folder.  

2. Run the program:
   ```bash
   python main.py
   ```

3. The webcam will open and receipts will be detected with bounding boxes.  

4. Press `ESC` to exit.  

---

## Notes

- If you have multiple cameras, change the index in `cv2.VideoCapture(0)`.  
- If you want to use a different model, change the path in `yolo/detection.py`.  
- If you get missing library errors, check your `requirements.txt`.  

---

## Contact

