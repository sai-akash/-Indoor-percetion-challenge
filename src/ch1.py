#!/usr/bin/env python3
"""
Challenge 1 – Automatic overlay for both office and bathroom
Generates detected PNGs and YAML metadata for each survey.
"""

import os, cv2, yaml, json, numpy as np
from pathlib import Path

def process_survey(survey, outputs_dir, gridmap_root, transform_file):
    print(f"\n[INFO] Processing survey: {survey}")

    # --- Select correct map filename
    map_name = "officeroom.yaml" if survey == "office" else "bathroom.yaml"
    yaml_path = gridmap_root / survey / map_name
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    map_img_path = gridmap_root / survey / meta["image"]
    if not map_img_path.exists():
        pgm = list((gridmap_root / survey).glob("*.pgm"))
        if pgm:
            map_img_path = pgm[0]
            print(f"[WARN] Using fallback image {map_img_path.name}")

    # --- Load map
    map_gray = cv2.imread(str(map_img_path), cv2.IMREAD_GRAYSCALE)
    map_color = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)
    h, w = map_gray.shape
    origin = meta["origin"][:2]
    resolution = float(meta["resolution"])

    print(f"[INFO] Map size: {w}x{h}, resolution: {resolution}m/px, origin: {origin}")

    # --- Alignment correction (none needed for Challenge 1)
    x_shift = 0
    y_shift = 0

    # --- Load detections
    det_path = outputs_dir / f"{survey}_detections.json"
    det = json.load(open(det_path))["detections"]
    print(f"[INFO] Loaded {len(det)} detections")

    yaml_out = []
    skipped_gray = 0
    skipped_bounds = 0

    for d in det:
        cls = d["class"]
        conf = d.get("confidence", 1.0)
        pos = d["pose"]["position"]
        dims = d.get("dimensions", {"width": 0.5, "height": 0.5})
        xw, yw = pos[0], pos[1]

        # Clamp realistic object sizes (0.3–1.0 m)
        width_m = max(0.3, min(dims.get("width", 0.5), 1.0))
        height_m = max(0.3, min(dims.get("height", 0.5), 1.0))
        w_px = int(width_m / resolution)
        h_px = int(height_m / resolution)

        # --- Use stored map coords if available, else compute
        if "map_coordinates" in d and d["map_coordinates"]:
            x_px = int(d["map_coordinates"]["x"])
            y_px = int(d["map_coordinates"]["y"])
        else:
            x_px = int((xw - origin[0]) / resolution + x_shift)
            y_px = int(((yw - origin[1]) / resolution) + y_shift)

        # --- Skip detections outside map bounds
        if not (0 <= x_px < w and 0 <= y_px < h):
            skipped_bounds += 1
            continue

        # --- Skip detections in gray/unknown area
        pixel_value = map_gray[y_px, x_px]
        
        # Check center pixel (gray is 200-210)
        if 200 <= pixel_value <= 210:
            skipped_gray += 1
            continue
        
        # Check surrounding region (5x5 pixels) to ensure not mostly gray
        check_radius = 5
        y_min = max(0, y_px - check_radius)
        y_max = min(h, y_px + check_radius)
        x_min = max(0, x_px - check_radius)
        x_max = min(w, x_px + check_radius)
        region = map_gray[y_min:y_max, x_min:x_max]
        mean_val = np.mean(region)
        
        if 200 <= mean_val <= 210:
            skipped_gray += 1
            continue

        # --- Draw boxes
        cv2.rectangle(map_color,
                      (x_px - w_px // 2, y_px - h_px // 2),
                      (x_px + w_px // 2, y_px + h_px // 2),
                      (0, 0, 255), 2)
        cv2.putText(map_color, cls,
                    (x_px - 10, y_px - h_px // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        yaml_out.append({
            "class": cls,
            "confidence": float(conf),
            "pose_map": [float(x_px), float(y_px)],
            "pose_world": [float(xw), float(yw)],
            "size_m": [float(width_m), float(height_m)]
        })

    print(f"[INFO] Skipped: {skipped_bounds} out-of-bounds, {skipped_gray} in gray/unmapped area")
    print(f"[INFO] Valid detections before NMS: {len(yaml_out)}")

    # -------------------------------------------------------------
    # Remove overlapping detections (IoU > 0.3)
    # -------------------------------------------------------------
    def iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    filtered, boxes = [], []
    for d in yaml_out:
        x, y = d["pose_map"]
        w_px = int(d["size_m"][0] / resolution)
        h_px = int(d["size_m"][1] / resolution)
        box = [x - w_px//2, y - h_px//2, x + w_px//2, y + h_px//2]
        if all(iou(box, b) <= 0.3 for b in boxes):
            filtered.append(d)
            boxes.append(box)
    
    print(f"[INFO] Valid detections after NMS: {len(filtered)}")
    yaml_out = filtered

    # --- Save results
    out_img = outputs_dir / f"{survey}_detected.png"
    out_yaml = outputs_dir / f"{survey}_detections_final.yaml"
    cv2.imwrite(str(out_img), map_color)
    yaml.dump(yaml_out, open(out_yaml, "w"))
    print(f"[✓] Saved {out_img.name} with {len(yaml_out)} detections")
    print(f"[✓] Saved {out_yaml.name}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
base = Path("~/aici_challenge").expanduser()
outputs = base / "outputs" / "challenge1"
gridmaps = base / "gridmap"
transform = base / "outputs" / "aligned_maps_transform.json"

print("="*60)
print("Challenge 1: Object Detection Overlay")
print("="*60)

for name in ["office", "bathroom"]:
    process_survey(name, outputs, gridmaps, transform)

print("\n" + "="*60)
print("✅ Challenge 1 completed for both surveys.")
print("="*60)
print("\nGenerated files:")
print("  - office_detected.png")
print("  - office_detections_final.yaml")
print("  - bathroom_detected.png")
print("  - bathroom_detections_final.yaml")