#!/usr/bin/env python3
"""
Object detection for bathroom and office datasets using YOLOv8.
Detects: Bathtub, Chair, Couch, Shelf, Table, WC (Toilet)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

# Classes to filter - only draw boxes for these
FILTER_CLASSES = ['Chair', 'Couch', 'Shelf', 'Table', 'WC', 'Bathtub']


def compute_iou(box1, box2):
    """Compute Intersection over Union between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.2):
    """Apply Non-Maximum Suppression to remove overlapping boxes.
    Uses very strict threshold to ensure NO overlapping boxes."""
    if len(detections) == 0:
        return []
    
    # Group by class
    by_class = {}
    for det in detections:
        cls = det['class_name']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)
    
    # Apply NMS per class
    keep_all = []
    for cls, dets in by_class.items():
        # Sort by confidence (highest first)
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(dets) > 0:
            # Keep the highest confidence detection
            best = dets.pop(0)
            keep.append(best)
            
            # Remove ALL overlapping detections (even slight overlap)
            remaining = []
            for det in dets:
                iou = compute_iou(best['bbox'], det['bbox'])
                # STRICT: Remove any box with even small overlap
                if iou < iou_threshold:
                    remaining.append(det)
            
            dets = remaining
        
        keep_all.extend(keep)
    
    return keep_all


def process_dataset(input_dir, output_dir, model, confidence=0.3, dataset_name=""):
    """Process all RGB images in a dataset directory."""
    
    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all RGB images
    rgb_images = sorted(input_path.glob("rgb_*.png"))
    
    if not rgb_images:
        print(f"[WARNING] No RGB images found in {input_dir}")
        return {}
    
    print(f"\n[INFO] Processing {len(rgb_images)} images from {input_dir}")
    
    detection_count = 0
    class_counts = {}
    is_bathroom = 'bathroom' in dataset_name.lower()
    
    for idx, img_path in enumerate(rgb_images):
        print(f"  [{idx+1}/{len(rgb_images)}] Processing {img_path.name}...", end=" ")
        
        try:
            # Run YOLO detection
            results = model(str(img_path), conf=confidence, verbose=False)
            result = results[0]
            
            # Get detections
            boxes = result.boxes
            raw_detections = []
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = model.names[cls_id]
                    
                    # Map to our target classes
                    mapped_name = None
                    
                    # Explicit mapping - ONLY these classes
                    if class_name == 'chair':
                        mapped_name = 'Chair'
                    elif class_name in ['couch', 'sofa']:
                        mapped_name = 'Couch'
                    elif class_name == 'dining table':
                        mapped_name = 'Table'
                    elif class_name == 'toilet':
                        mapped_name = 'WC'
                    elif class_name == 'book' and not is_bathroom:
                        # Books might indicate shelves, but only in office
                        mapped_name = 'Shelf'
                    
                    # For bathroom: aggressive bathtub detection
                    if is_bathroom and class_name in ['bed', 'couch', 'sink']:
                        # Check bounding box characteristics
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        area = width * height
                        
                        # Bathtubs are typically:
                        # - Large (area > 30000 pixels)
                        # - Wide and low (aspect_ratio > 1.2)
                        # - Located low in image (y2 > image_height * 0.3)
                        if area > 30000 and aspect_ratio > 1.2:
                            mapped_name = 'Bathtub'
                    
                    # Only keep if mapped to our target classes
                    if mapped_name is not None and mapped_name in FILTER_CLASSES:
                        raw_detections.append({
                            'class_name': mapped_name,
                            'confidence': conf,
                            'bbox': xyxy.tolist()
                        })
            
            # Apply Non-Maximum Suppression with VERY strict threshold
            filtered_detections = non_max_suppression(raw_detections, iou_threshold=0.2)
            
            detection_count += len(filtered_detections)
            
            if len(filtered_detections) > 0:
                print(f"✓ Found {len(filtered_detections)} objects")
            else:
                print("✓ No target objects detected")
            
            # Read image for drawing
            img = cv2.imread(str(img_path))
            
            # Draw boxes
            for det in filtered_detections:
                bbox = [int(x) for x in det['bbox']]
                x1, y1, x2, y2 = bbox
                
                # Different colors for different classes
                color_map = {
                    'Chair': (0, 255, 0),      # Green
                    'Couch': (255, 0, 0),       # Blue
                    'Table': (0, 255, 255),     # Yellow
                    'WC': (255, 0, 255),        # Magenta
                    'Bathtub': (0, 165, 255),   # Orange
                    'Shelf': (255, 255, 0)      # Cyan
                }
                color = color_map.get(det['class_name'], (0, 255, 0))
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(img, (x1, y1 - label_h - 10), 
                            (x1 + label_w + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Count detections
                class_counts[det['class_name']] = class_counts.get(det['class_name'], 0) + 1
            
            # Save result
            output_file = output_path / f"detected_{img_path.name}"
            cv2.imwrite(str(output_file), img)
            
            # Save detection metadata
            meta_file = output_path / f"detections_{img_path.stem}.txt"
            with open(meta_file, 'w') as f:
                for det in filtered_detections:
                    bbox = det['bbox']
                    f.write(f"{det['class_name']},{det['confidence']:.3f},"
                           f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}\n")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[SUMMARY] Total detections: {detection_count}")
    print(f"[SUMMARY] Detections by class:")
    for cls_name in FILTER_CLASSES:
        count = class_counts.get(cls_name, 0)
        if count > 0:
            print(f"  - {cls_name}: {count}")
    print(f"[SUMMARY] Output saved to: {output_dir}")
    
    return class_counts


def main():
    parser = argparse.ArgumentParser(
        description="Object detection for bathroom and office datasets using YOLOv8"
    )
    parser.add_argument(
        "--bathroom",
        default="~/aici_challenge/outputs/bathroom",
        help="Path to bathroom dataset"
    )
    parser.add_argument(
        "--office",
        default="~/aici_challenge/outputs/office",
        help="Path to office dataset"
    )
    parser.add_argument(
        "--out-bathroom",
        default="~/aici_challenge/outputs/bathroom_detected",
        help="Output path for bathroom detections"
    )
    parser.add_argument(
        "--out-office",
        default="~/aici_challenge/outputs/office_detected",
        help="Output path for office detections"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1). Recommended: 0.25-0.5"
    )
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model size"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Object Detection with YOLOv8")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS IoU threshold: 0.2 (strict - NO overlaps)")
    print(f"Target objects: {', '.join(FILTER_CLASSES)}")
    print("NOTE: Company requirement - NO overlapping bounding boxes")
    print("="*60)
    
    # Load YOLO model
    print("\n[INFO] Loading YOLOv8 model...")
    model = YOLO(args.model)
    model.to(args.device)
    print(f"[INFO] Model loaded successfully ({len(model.names)} classes)")
    
    total_bathroom = {}
    total_office = {}
    
    # Process bathroom dataset
    if os.path.exists(os.path.expanduser(args.bathroom)):
        print("\n" + "="*60)
        print("BATHROOM DATASET")
        print("="*60)
        total_bathroom = process_dataset(
            args.bathroom,
            args.out_bathroom,
            model,
            args.confidence,
            dataset_name="bathroom"
        )
    else:
        print(f"\n[WARNING] Bathroom dataset not found: {args.bathroom}")
    
    # Process office dataset
    if os.path.exists(os.path.expanduser(args.office)):
        print("\n" + "="*60)
        print("OFFICE DATASET")
        print("="*60)
        total_office = process_dataset(
            args.office,
            args.out_office,
            model,
            args.confidence,
            dataset_name="office"
        )
    else:
        print(f"\n[WARNING] Office dataset not found: {args.office}")
    
    # Combined summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    all_classes = [c for c in FILTER_CLASSES if c in set(list(total_bathroom.keys()) + list(total_office.keys()))]
    
    if all_classes:
        print(f"\n{'Class':<20} {'Bathroom':<12} {'Office':<12} {'Total':<12}")
        print("-" * 60)
        for cls_name in all_classes:
            bath_count = total_bathroom.get(cls_name, 0)
            office_count = total_office.get(cls_name, 0)
            total = bath_count + office_count
            print(f"{cls_name:<20} {bath_count:<12} {office_count:<12} {total:<12}")
    else:
        print("\nNo detections found in either dataset.")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()