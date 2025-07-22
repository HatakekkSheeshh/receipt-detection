import cv2
from ultralytics import YOLO

def detection():
    # Load YOLOv8 model
    model = YOLO('best.pt')
    names = model.names

    # Open video
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                label = names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1 + 3, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
    cap.release()
    cv2.destroyAllWindows()

