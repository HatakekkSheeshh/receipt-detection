import os
import sys
from yolo.detection import detection

def run():
    detection()

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"Error: {e}")
