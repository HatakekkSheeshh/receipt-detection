import cv2
from ultralytics import YOLO
from yolo.utils import detect_objects, print_detected_objects
from yolo.types import DetectedObject
from ocr.processing import process_ocr
from easyocr import Reader

def detection():
    # Create model
    model = YOLO('./data/best.pt')
    names = model.names
    print(f"Model classes: {names}")

    # Open video
    cap = cv2.VideoCapture(0)
    frame_count = 0
    reader = Reader(['en', 'vi'], gpu=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        detected_objects = detect_objects(frame, model)

        # Draw bounding boxes and labels
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            label = obj.label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 3, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("RGB", frame)
        
        if detected_objects:
            print_detected_objects(detected_objects)
            ocr_result = process_ocr(frame=frame, detected_objects=detected_objects, reader=reader)
            print(ocr_result)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()





