#!/usr/bin/env python3
"""
OCR Performance Benchmark with PaddleOCR SVTR
Uses DBNet for detection and SVTR for recognition
"""

from pathlib import Path
import json
import time
import sys
import os
import numpy as np
from statistics import mean, stdev
from PIL import Image
from paddleocr import PaddleOCR

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from yolo_detect_bill.bill_detector import BillDetector
import cv2

class OCRBenchmark:
    """OCR performance benchmark with DBNet + SVTR"""
    
    def __init__(self):
        # Initialize YOLO
        yolo_model_path = Path(__file__).parent.parent / "yolo_detect_bill" / "bill_models.pt"
        self.yolo_detector = BillDetector(model_path=str(yolo_model_path))
        self.yolo_detector.load_model()
        
        # Initialize engines
        src_folder = Path(__file__).parent
        project_root = src_folder.parent
        det_model_path = str(project_root / "dbnet" / "model")
        rec_model_path = str(project_root / "svtr" / "model")

        # Detection-only engine (DBNet)
        print("🔧 Loading DBNet detection engine...")
        self.det_engine = PaddleOCR(
            det_model_dir=det_model_path,
            rec=False,  # Disable recognition
            use_angle_cls=False,
            use_gpu=False,
            show_log=False
        )
        
        # Recognition with SVTR
        print("🔧 Loading SVTR recognition engine...")
        try:
            # SVTR model configuration
            # SVTR is more robust for scene text and supports various text orientations
            self.svtr_engine = PaddleOCR(
                det=False,  # Disable detection
                rec=True,   # Enable recognition only
                use_angle_cls=False,
                use_gpu=False,
                lang='en',
                rec_model_dir=rec_model_path,  # Use default SVTR model
                rec_algorithm='SVTR_LCNet',  # Specify SVTR algorithm
                show_log=False
            )
            print("✅ Manual DBNet+SVTR pipeline loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load SVTR model: {e}")
            print("Falling back to default PaddleOCR recognition...")
            self.svtr_engine = PaddleOCR(
                det=False,
                rec=True,
                use_angle_cls=False,
                use_gpu=False,
                lang='en',
                show_log=False
            )

        # Initialize baseline PaddleOCR
        print("🔧 Loading baseline PaddleOCR...")
        self.baseline_engine = PaddleOCR(
            use_angle_cls=False,
            use_gpu=False,
            lang='en',
            show_log=False
        )
        print("✅ Baseline PaddleOCR loaded successfully!")
    
    def crop_bill_region(self, image, confidence_threshold=0.1):
        """Crop best bill region from image"""
        detections = self.yolo_detector.detect_bills_from_frame(image, confidence_threshold)
        
        if not detections:
            return None
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        x1, y1, x2, y2 = (
            int(best_detection['x1']), int(best_detection['y1']),
            int(best_detection['x2']), int(best_detection['y2'])
        )
        
        # Add padding
        padding = 10
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def preprocess_image(self, image, min_height=64, min_width=64, max_height=1024, max_width=1024):
        """Preprocess image for OCR detection"""
        h, w = image.shape[:2]
        
        # Apply light enhancement if needed
        processed_image = image.copy()
        
        # Only apply enhancement if image is very dark/low contrast
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80 or mean_brightness > 220:
            # Apply mild contrast enhancement
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
       
        # Resize if image is too small
        if h < min_height or w < min_width:
            scale_h = min_height / h if h < min_height else 1.0
            scale_w = min_width / w if w < min_width else 1.0
            scale = max(scale_h, scale_w)
            
            new_h = max(int(h * scale), min_height)
            new_w = max(int(w * scale), min_width)
            
            processed_image = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"   🔍 Upscaled from {w}x{h} to {new_w}x{new_h}")
        
        # Resize if image is too large
        elif h > max_height or w > max_width:
            scale_h = max_height / h if h > max_height else 1.0
            scale_w = max_width / w if w > max_width else 1.0
            scale = min(scale_h, scale_w)
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            processed_image = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"   🔍 Downscaled from {w}x{h} to {new_w}x{new_h}")
        
        return processed_image

    def dbnet_postprocess(self, image, text_box, min_width=16, min_height=12, max_aspect_ratio=25):
        """Crop text regions from DBNet output for SVTR processing"""
        # DBNet text_box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        points = np.array(text_box, dtype=np.float32).reshape(-1, 2)
        
        # Get bounding rectangle from the 4 corner points
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        
        # SVTR-specific size filtering
        # SVTR works better with slightly larger minimum dimensions
        if w < min_width or h < min_height:
            return None
        
        # SVTR handles moderate aspect ratios well
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > max_aspect_ratio:
            return None
        
        # Size limit for SVTR processing
        if w > 1500 or h > 800:
            return None
        
        # Add padding for SVTR (needs more context)
        padding = 8
        img_h, img_w = image.shape[:2]
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        
        # Ensure minimum size after padding
        final_w = x2 - x1
        final_h = y2 - y1
        
        if final_w < min_width or final_h < min_height:
            return None
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return None
            
        return cropped
    
    def svtr_preprocess(self, image):
        """Preprocess cropped text region for SVTR input"""
        if image is None or image.size == 0:
            return None
            
        h, w = image.shape[:2]
        
        # Minimum size check for SVTR
        if h < 12 or w < 16:
            return None
        
        processed = image.copy()
        
        # SVTR-specific preprocessing
        # 1. Ensure adequate height for SVTR (works best with height >= 32)
        target_height = 48  # SVTR optimal height
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w
        
        # 2. Apply contrast enhancement for better SVTR recognition
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # More aggressive enhancement for SVTR
        if mean_brightness < 100 or mean_brightness > 180:
            # Apply CLAHE for better text contrast
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # 3. Handle extreme aspect ratios for SVTR
        aspect_ratio = w / h
        if aspect_ratio > 20:  # Very wide text
            # Slightly reduce width to improve recognition
            new_w = min(w, h * 18)
            if new_w != w:
                processed = cv2.resize(processed, (new_w, h), interpolation=cv2.INTER_AREA)
        
        return processed
    
    def ocr_pipeline(self, image):
        """Manual pipeline: DBNet detection -> SVTR recognition"""
        # Step 1: Detection with DBNet
        print("   🔍 Running DBNet detection...")
        det_result = self.det_engine.ocr(image, rec=False)
        
        if not det_result or not det_result[0]:
            return {
                'results': [],
                'text_count': 0,
                'avg_confidence': 0,
                'high_confidence_count': 0,
                'confidences': []
            }
        
        detected_boxes = det_result[0]
        print(f"   🔍 DBNet detected {len(detected_boxes)} text regions")
        
        # Step 2: Recognition with SVTR
        recognized_results = []
        confidences = []
        
        for i, text_box in enumerate(detected_boxes):
            # Crop text region from DBNet output
            text_region = self.dbnet_postprocess(image, text_box)
            if text_region is None:
                continue
            
            # Preprocess for SVTR
            svtr_input = self.svtr_preprocess(text_region)
            if svtr_input is None:
                continue
            
            try:
                # Recognition with SVTR
                # Use recognition-only mode
                svtr_result = self.svtr_engine.ocr(svtr_input, det=False, rec=True, cls=False)
                
                if svtr_result and len(svtr_result) > 0 and svtr_result[0] is not None:
                    for line in svtr_result[0]:
                        if line is not None and len(line) >= 2:
                            # SVTR returns [text, confidence] format in rec-only mode
                            text_info = line
                            if len(text_info) >= 2:
                                text = str(text_info[0]) if text_info[0] is not None else ""
                                confidence = float(text_info[1]) if text_info[1] is not None else 0.0
                                
                                # Filter results
                                if confidence > 0.2 and text.strip() and len(text.strip()) >= 1:
                                    recognized_results.append({
                                        'coords': text_box,  # Use original DBNet coordinates
                                        'text': text.strip(),
                                        'confidence': confidence
                                    })
                                    confidences.append(confidence)
                                    break  # Take only the best result per region
                            
            except Exception as e:
                print(f"   ⚠️ SVTR recognition failed for region {i}: {e}")
                continue
        
        # Calculate metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence_count = sum(1 for c in confidences if c > 0.9)
        
        return {
            'results': recognized_results,
            'text_count': len(recognized_results),
            'avg_confidence': avg_confidence,
            'high_confidence_count': high_confidence_count,
            'confidences': confidences
        }

    def process_image(self, image, engine_name="custom"):
        """Process image with specified OCR engine"""
        if engine_name == "custom":
            processed_image = self.preprocess_image(image)
            return self.ocr_pipeline(processed_image)
            
        elif engine_name == "baseline":
            processed_image = self.preprocess_image(image)
            result = self.baseline_engine.ocr(processed_image, cls=False)
            
            if result is None or not isinstance(result, list) or len(result) == 0:
                result = [[]]
            elif result[0] is None:
                result = [[]]
            
            # Parse baseline results
            parsed_results = []
            confidences = []

            if result and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    if line is not None and len(line) >= 2:
                        coords = line[0]
                        text_info = line[1]
                        
                        if text_info is not None and len(text_info) >= 2:
                            text = str(text_info[0]) if text_info[0] is not None else ""
                            confidence = float(text_info[1]) if text_info[1] is not None else 0.0
                            
                            if confidence > 0.1 and text.strip() and len(text.strip()) > 0:
                                parsed_results.append({
                                    'coords': coords,
                                    'text': text.strip(),
                                    'confidence': confidence
                                })
                                confidences.append(confidence)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            high_confidence_count = sum(1 for c in confidences if c > 0.9)
            
            return {
                'results': parsed_results,
                'text_count': len(parsed_results),
                'avg_confidence': avg_confidence,
                'high_confidence_count': high_confidence_count,
                'confidences': confidences
            }

    def run_benchmark(self, num_images=10):
        """Run benchmark on multiple images"""
        print("🏁 OCR Performance Benchmark - SVTR Recognition")
        print("=" * 60)

        # Get test images
        image_folder = Path(__file__).parent.parent / "image_test"
        test_images = list(image_folder.glob("*.jpg"))[:num_images]
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        print(f"📸 Testing {len(test_images)} images...")
        
        ocr_results = []
        baseline_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n🔄 Processing {i}/{len(test_images)}: {image_path.name}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path.name}")
                continue

            bill_crop = self.crop_bill_region(image)
            
            if bill_crop is None:
                print(f"⚠️ No bill detected in {image_path.name}")
                continue

            # Process with custom engine
            print("   🔧 Processing with Custom DBNet+SVTR...")
            start_time = time.time()
            ocr_result = self.process_image(bill_crop, "custom")
            ocr_time = time.time() - start_time

            print("   🔧 Processing with Baseline PaddleOCR...")
            start_time = time.time()
            baseline_result = self.process_image(bill_crop, "baseline")
            baseline_time = time.time() - start_time    
            
            # Store results
            ocr_results.append({
                'image': image_path.name,
                'processing_time': ocr_time,
                'text_count': ocr_result['text_count'],
                'avg_confidence': ocr_result['avg_confidence'],
                'high_confidence_count': ocr_result['high_confidence_count'],
                'results': ocr_result['results']
            })
            
            baseline_results.append({
                'image': image_path.name,
                'processing_time': baseline_time,
                'text_count': baseline_result['text_count'],
                'avg_confidence': baseline_result['avg_confidence'],
                'high_confidence_count': baseline_result['high_confidence_count'],
                'results': baseline_result['results']
            })

            print(f"   ✅ Custom:   {ocr_result['text_count']} texts in {ocr_time:.2f}s")
            print(f"   ✅ Baseline: {baseline_result['text_count']} texts in {baseline_time:.2f}s")
                
        if baseline_results:
            print(f"\n📊 Generating report...")
            self.generate_report(ocr_results, baseline_results)
        else:
            print("❌ No images were successfully processed!")
    
    def generate_report(self, ocr_results, baseline_results):
        """Generate benchmark report"""
        print(f"\n📊 BENCHMARK REPORT - SVTR Recognition")
        print("=" * 60)

        baseline_times = [r['processing_time'] for r in baseline_results if r['processing_time'] > 0]
        baseline_counts = [r['text_count'] for r in baseline_results]
        baseline_confs = [r['avg_confidence'] for r in baseline_results if r['avg_confidence'] > 0]

        print(f"⏱️  BASELINE PROCESSING TIME: avg={mean(baseline_times):.3f}s")
        print(f"📝 BASELINE TEXTS RECOGNIZED: avg={mean(baseline_counts):.1f}")
        if baseline_confs:
            print(f"🎯 BASELINE CONFIDENCE: {mean(baseline_confs):.3f}")

        if ocr_results and any(r['text_count'] > 0 for r in ocr_results):
            ocr_times = [r['processing_time'] for r in ocr_results if r['processing_time'] > 0]
            ocr_counts = [r['text_count'] for r in ocr_results]
            ocr_confs = [r['avg_confidence'] for r in ocr_results if r['avg_confidence'] > 0]
            
            if ocr_times:
                print(f"⏱️  CUSTOM PROCESSING TIME: avg={mean(ocr_times):.3f}s")
                print(f"📝 CUSTOM TEXTS RECOGNIZED: avg={mean(ocr_counts):.1f}")
            if ocr_confs:
                print(f"🎯 CUSTOM CONFIDENCE: {mean(ocr_confs):.3f}")

        print(f"\n🎉 SVTR Benchmark completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OCR Performance Benchmark with SVTR')
    parser.add_argument('--images', type=int, default=10, help='Number of images to test')
    args = parser.parse_args()
    
    benchmark = OCRBenchmark()
    benchmark.run_benchmark(num_images=args.images)