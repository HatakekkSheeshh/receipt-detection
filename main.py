from huggingface_hub import snapshot_download
import os
import sys
from yolo.detection import detection

if __name__ == '__main__':
    try:
        detection()
    except Exception as e:
        print(f"Error: {e}")
