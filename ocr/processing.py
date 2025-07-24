from easyocr import Reader

def process_ocr(frame, detected_objects, reader: Reader):
    ocr_results = []
    for obj in detected_objects:
        x1, y1, x2, y2 = obj.bbox
        roi = frame[y1:y2, x1:x2]
        result = reader.readtext(roi, detail=0)  # detail=0: take text, detail=1: take box and score
        text = ' '.join(result)
        ocr_results.append({
            'track_id': obj.track_id,
            'label': obj.label,
            'bbox': obj.bbox,
            'confidence': obj.confidence,
            'text': text
        })
    return ocr_results