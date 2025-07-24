class DetectedObject:
    def __init__(self, track_id, bbox, label, class_id, confidence):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.class_id = class_id
        self.confidence = confidence
    
    def __str__(self):
        return f"Object(ID:{self.track_id}, Label:{self.label}, BBox:{self.bbox}, Conf:{self.confidence:.2f})"
    
    def __repr__(self):
        return self.__str__()
    
    # Convert to json
    def to_dict(self):
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'label': self.label,
            'class_id': self.class_id,
            'confidence': self.confidence
        }
