from yolo.types import DetectedObject

def detect_objects(frame, model):
    """
    Function to detect objects in a frame and return bounding boxes
    """
    results = model.track(frame, persist=True)
    detected_objects = []
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            label = model.names[class_id]
            
            obj = DetectedObject(
                track_id=int(track_id),
                bbox=[x1, y1, x2, y2],
                label=label,
                class_id=int(class_id),
                confidence=float(results[0].boxes.conf[class_ids.index(class_id)])
            )
            
            detected_objects.append(obj)
    

    return detected_objects

def print_detected_objects(detected_objects):
    """
    Print detected objects in a readable format
    """
    if not detected_objects:
        print("No objects detected")
        return
    
    print(f"Detected {len(detected_objects)} objects:")
    for i, obj in enumerate(detected_objects, 1):
        print(f"  {i}. {obj}")