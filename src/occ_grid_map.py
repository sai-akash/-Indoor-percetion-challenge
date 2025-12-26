#!/usr/bin/env python3
"""
Improved Occupancy Grid Map Generator with Detection Clustering
- Removes duplicate detections across frames
- Clusters nearby detections into single objects
- Creates clean visualizations
"""

import os
import cv2
import numpy as np
import json
import yaml
from pathlib import Path
import argparse
from collections import defaultdict
from sklearn.cluster import DBSCAN

def load_camera_intrinsics(data_dir):
    """Load camera intrinsics from npz file."""
    intrinsics_file = Path(data_dir) / "camera_intrinsics.npz"
    if intrinsics_file.exists():
        data = np.load(intrinsics_file)
        return data['K'], data['D']
    return None, None

def load_detection_file(det_file):
    """Load detections from text file."""
    detections = []
    if det_file.exists():
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 6:
                    detections.append({
                        'class': parts[0],
                        'confidence': float(parts[1]),
                        'bbox': [float(parts[2]), float(parts[3]), 
                                float(parts[4]), float(parts[5])]
                    })
    return detections

def depth_to_point_cloud(depth_img, K, max_depth=10000.0, downsample=4):
    """Convert depth image to 3D point cloud with downsampling."""
    h, w = depth_img.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    depth_downsampled = depth_img[::downsample, ::downsample]
    h_down, w_down = depth_downsampled.shape
    
    u, v = np.meshgrid(np.arange(w_down), np.arange(h_down))
    u = u * downsample
    v = v * downsample
    
    valid = (depth_downsampled > 0) & (depth_downsampled < max_depth)
    
    z = depth_downsampled[valid] / 1000.0  # mm to meters
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    return points

def project_to_2d_map(points, tf_matrix, resolution=0.05):
    """Project 3D points to 2D occupancy grid."""
    ones = np.ones((points.shape[0], 1))
    points_homog = np.hstack([points, ones])
    points_world = (tf_matrix @ points_homog.T).T[:, :3]
    
    xy = points_world[:, :2]
    return xy

def create_occupancy_grid(all_points, resolution=0.05):
    """Create occupancy grid from accumulated points."""
    if len(all_points) == 0:
        print("[WARNING] No points to create occupancy grid")
        return np.zeros((500, 500), dtype=np.uint8), (0, 0), resolution
    
    all_points = [pts for pts in all_points if len(pts) > 0]
    
    if len(all_points) == 0:
        print("[WARNING] All point arrays are empty")
        return np.zeros((500, 500), dtype=np.uint8), (0, 0), resolution
    
    xy = np.vstack(all_points)
    
    if xy.shape[0] == 0:
        print("[WARNING] No valid points after stacking")
        return np.zeros((500, 500), dtype=np.uint8), (0, 0), resolution
    
    min_x, min_y = xy.min(axis=0)
    max_x, max_y = xy.max(axis=0)
    
    padding = 2.0
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    
    grid = np.zeros((height, width), dtype=np.uint8)
    
    for points in all_points:
        if len(points) == 0:
            continue
        grid_x = ((points[:, 0] - min_x) / resolution).astype(int)
        grid_y = ((points[:, 1] - min_y) / resolution).astype(int)
        
        valid = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)
        grid[grid_y[valid], grid_x[valid]] = 255
    
    origin = (min_x, min_y)
    return grid, origin, resolution

def world_to_grid(world_pos, origin, resolution):
    """Convert world coordinates to grid coordinates."""
    grid_x = int((world_pos[0] - origin[0]) / resolution)
    grid_y = int((world_pos[1] - origin[1]) / resolution)
    return grid_x, grid_y

def estimate_3d_bbox(bbox_2d, depth_img, K, tf_matrix):
    """Estimate 3D bounding box from 2D detection and depth."""
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    h, w = depth_img.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    depth_roi = depth_img[y1:y2, x1:x2]
    valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < 10000)]
    
    if len(valid_depths) == 0:
        return None
    
    median_depth = np.median(valid_depths) / 1000.0
    
    center_u = (x1 + x2) / 2
    center_v = (y1 + y2) / 2
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_cam = (center_u - cx) * median_depth / fx
    y_cam = (center_v - cy) * median_depth / fy
    z_cam = median_depth
    
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    point_world = tf_matrix @ point_cam
    
    width_pixels = x2 - x1
    height_pixels = y2 - y1
    
    width_3d = width_pixels * median_depth / fx
    height_3d = height_pixels * median_depth / fy
    
    return {
        'center': point_world[:3].tolist(),
        'dimensions': [width_3d, height_3d, height_3d * 0.8],
        'depth': float(median_depth)
    }

def compute_overlap_2d(det1, det2):
    """Compute 2D overlap between two detections based on their positions and dimensions."""
    pos1 = det1['pose']['position'][:2]
    pos2 = det2['pose']['position'][:2]
    
    w1 = det1['dimensions']['width']
    h1 = det1['dimensions']['height']
    w2 = det2['dimensions']['width']
    h2 = det2['dimensions']['height']
    
    # Bounding box corners
    x1_min, y1_min = pos1[0] - w1/2, pos1[1] - h1/2
    x1_max, y1_max = pos1[0] + w1/2, pos1[1] + h1/2
    
    x2_min, y2_min = pos2[0] - w2/2, pos2[1] - h2/2
    x2_max, y2_max = pos2[0] + w2/2, pos2[1] + h2/2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def remove_overlapping_detections(detections, iou_threshold=0.3):
    """Remove overlapping detections across ALL classes.
    Keeps highest confidence detection when boxes overlap."""
    
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for det in detections:
        # Check if this detection overlaps with any kept detection
        overlaps = False
        for kept_det in keep:
            overlap = compute_overlap_2d(det, kept_det)
            if overlap > iou_threshold:
                overlaps = True
                break
        
        if not overlaps:
            keep.append(det)
    
    removed = len(detections) - len(keep)
    if removed > 0:
        print(f"[INFO] Removed {removed} overlapping detections (IoU > {iou_threshold})")
    
    return keep


def cluster_detections(all_detections_3d, distance_threshold=0.8):
    """Cluster detections across frames to remove duplicates.
    Objects detected in multiple frames are merged into one.
    Uses aggressive threshold to handle dense environments."""
    
    if len(all_detections_3d) == 0:
        return []
    
    # Group by class
    by_class = defaultdict(list)
    for det in all_detections_3d:
        by_class[det['class']].append(det)
    
    clustered_detections = []
    
    for cls, dets in by_class.items():
        if len(dets) == 0:
            continue
        
        # Get positions (x, y only - 2D clustering)
        positions = np.array([d['pose']['position'][:2] for d in dets])
        
        # Use DBSCAN with larger epsilon for aggressive clustering
        # min_samples=1 means every point forms a cluster
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Merge detections in each cluster
        for label in set(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_dets = [dets[i] for i in cluster_indices]
            
            # Take detection with highest confidence
            best_det = max(cluster_dets, key=lambda x: x['confidence'])
            
            # Use median position instead of mean (more robust)
            median_position = np.median([d['pose']['position'] for d in cluster_dets], axis=0)
            best_det['pose']['position'] = median_position.tolist()
            
            # Use max dimensions to ensure object is fully covered
            max_width = max([d['dimensions']['width'] for d in cluster_dets])
            max_height = max([d['dimensions']['height'] for d in cluster_dets])
            best_det['dimensions']['width'] = float(max_width)
            best_det['dimensions']['height'] = float(max_height)
            
            clustered_detections.append(best_det)
    
    print(f"[INFO] Clustered {len(all_detections_3d)} detections → {len(clustered_detections)} unique objects")
    return clustered_detections
    """Cluster detections across frames to remove duplicates.
    Objects detected in multiple frames are merged into one.
    Uses aggressive threshold to handle dense environments."""
    
    if len(all_detections_3d) == 0:
        return []
    
    # Group by class
    by_class = defaultdict(list)
    for det in all_detections_3d:
        by_class[det['class']].append(det)
    
    clustered_detections = []
    
    for cls, dets in by_class.items():
        if len(dets) == 0:
            continue
        
        # Get positions (x, y only - 2D clustering)
        positions = np.array([d['pose']['position'][:2] for d in dets])
        
        # Use DBSCAN with larger epsilon for aggressive clustering
        # min_samples=1 means every point forms a cluster
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        # Merge detections in each cluster
        for label in set(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_dets = [dets[i] for i in cluster_indices]
            
            # Take detection with highest confidence
            best_det = max(cluster_dets, key=lambda x: x['confidence'])
            
            # Use median position instead of mean (more robust)
            median_position = np.median([d['pose']['position'] for d in cluster_dets], axis=0)
            best_det['pose']['position'] = median_position.tolist()
            
            # Use max dimensions to ensure object is fully covered
            max_width = max([d['dimensions']['width'] for d in cluster_dets])
            max_height = max([d['dimensions']['height'] for d in cluster_dets])
            best_det['dimensions']['width'] = float(max_width)
            best_det['dimensions']['height'] = float(max_height)
            
            clustered_detections.append(best_det)
    
    print(f"[INFO] Clustered {len(all_detections_3d)} detections → {len(clustered_detections)} unique objects")
    return clustered_detections

def draw_detection_on_grid(grid_img, detection, origin, resolution, color, label):
    """Draw oriented bounding box on grid map."""
    center = detection['pose']['position'][:2]
    width = detection['dimensions']['width']
    height = detection['dimensions']['height']
    
    half_w, half_h = width / 2, height / 2
    corners = [
        [center[0] - half_w, center[1] - half_h],
        [center[0] + half_w, center[1] - half_h],
        [center[0] + half_w, center[1] + half_h],
        [center[0] - half_w, center[1] + half_h]
    ]
    
    grid_corners = []
    for corner in corners:
        gx, gy = world_to_grid(corner, origin, resolution)
        grid_corners.append([gx, gy])
    
    grid_corners = np.array(grid_corners, dtype=np.int32)
    
    # Draw filled rectangle with transparency
    overlay = grid_img.copy()
    cv2.fillPoly(overlay, [grid_corners], color)
    cv2.addWeighted(overlay, 0.3, grid_img, 0.7, 0, grid_img)
    
    # Draw outline
    cv2.polylines(grid_img, [grid_corners], True, color, 3)
    
    # Draw label
    text_pos = tuple(grid_corners[0])
    cv2.putText(grid_img, label, text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

def process_survey(input_dir, output_dir, dataset_name):
    """Process a complete survey."""
    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} Survey")
    print(f"{'='*60}")
    
    K, D = load_camera_intrinsics(input_path)
    if K is None:
        print("[ERROR] Could not load camera intrinsics")
        return
    
    rgb_files = sorted(input_path.glob("rgb_*.png"))
    depth_files = sorted(input_path.glob("depth_raw_*.npy"))
    tf_files = sorted(input_path.glob("tf_odom_to_cam_*.npy"))
    
    print(f"[INFO] Found {len(rgb_files)} RGB, {len(depth_files)} depth, {len(tf_files)} TF files")
    
    all_points_2d = []
    all_detections_3d = []
    
    color_map = {
        'Chair': (0, 255, 0),
        'Couch': (255, 0, 0),
        'Table': (0, 255, 255),
        'WC': (255, 0, 255),
        'Bathtub': (0, 165, 255)
    }
    
    # Process frames (skip every 5th for memory)
    frame_skip = 5
    total_frames = min(len(rgb_files), len(depth_files), len(tf_files))
    frames_to_process = range(0, total_frames, frame_skip)
    
    print(f"[INFO] Processing {len(frames_to_process)} frames (every {frame_skip}th)")
    
    for frame_idx, idx in enumerate(frames_to_process):
        if frame_idx % 20 == 0:
            print(f"  [{frame_idx+1}/{len(frames_to_process)}] Processing frame {idx:05d}...")
        
        try:
            depth_raw = np.load(depth_files[idx])
            tf_matrix = np.load(tf_files[idx])
            
            if depth_raw is None or depth_raw.size == 0:
                continue
            
            points_3d = depth_to_point_cloud(depth_raw, K, max_depth=10000.0, downsample=4)
            
            if len(points_3d) == 0:
                continue
            
            points_2d = project_to_2d_map(points_3d, tf_matrix)
            
            if len(points_2d) > 0:
                all_points_2d.append(points_2d)
            
            del points_3d, points_2d
            
            # Load detections
            det_file = output_path / f"detections_rgb_{idx:05d}.txt"
            detections = load_detection_file(det_file)
            
            for det in detections:
                bbox_3d = estimate_3d_bbox(det['bbox'], depth_raw, K, tf_matrix)
                
                if bbox_3d is not None:
                    all_detections_3d.append({
                        'frame': idx,
                        'class': det['class'],
                        'confidence': det['confidence'],
                        'pose': {
                            'position': bbox_3d['center'],
                            'orientation': [0, 0, 0, 1]
                        },
                        'dimensions': {
                            'width': bbox_3d['dimensions'][0],
                            'height': bbox_3d['dimensions'][1],
                            'depth': bbox_3d['dimensions'][2]
                        },
                        'bbox_2d': det['bbox']
                    })
            
            del depth_raw
        
        except Exception as e:
            print(f"  [ERROR] Frame {idx}: {e}")
            continue
    
    print("\n[INFO] Creating occupancy grid map...")
    
    if len(all_points_2d) == 0:
        print("[ERROR] No point cloud data available")
        return
    
    grid, origin, resolution = create_occupancy_grid(all_points_2d, resolution=0.05)
    
    # Cluster detections to remove duplicates (VERY aggressive for dense environments)
    print("\n[INFO] Clustering detections...")
    clustered_detections = cluster_detections(all_detections_3d, distance_threshold=1.5)
    
    # Remove overlapping detections across different classes (very strict)
    print("[INFO] Removing overlapping detections across classes...")
    final_detections = remove_overlapping_detections(clustered_detections, iou_threshold=0.05)
    
    # Convert to color image
    grid_color = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    
    # Draw final detections
    print(f"[INFO] Drawing {len(final_detections)} unique objects on map...")
    for det in final_detections:
        color = color_map.get(det['class'], (128, 128, 128))
        label = f"{det['class']}"
        draw_detection_on_grid(grid_color, det, origin, resolution, color, label)
    
    # Save map
    output_img = output_path / f"{dataset_name}_occupancy_grid_with_detections.png"
    cv2.imwrite(str(output_img), grid_color)
    print(f"[INFO] Saved: {output_img}")
    
    # Save JSON
    output_json = output_path / f"{dataset_name}_detections.json"
    detection_data = {
        'survey': dataset_name,
        'resolution': resolution,
        'origin': origin,
        'detections': final_detections,
        'total_detections': len(final_detections),
        'classes': list(set([d['class'] for d in final_detections]))
    }
    
    with open(output_json, 'w') as f:
        json.dump(detection_data, f, indent=2)
    print(f"[INFO] Saved: {output_json}")
    
    # Save YAML
    output_yaml = output_path / f"{dataset_name}_detections.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump(detection_data, f, default_flow_style=False)
    print(f"[INFO] Saved: {output_yaml}")
    
    # Summary
    class_counts = defaultdict(int)
    for det in final_detections:
        class_counts[det['class']] += 1
    
    print(f"\n[SUMMARY] {dataset_name.upper()} - Unique Objects:")
    for cls, count in sorted(class_counts.items()):
        print(f"  - {cls}: {count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bathroom-input", default="~/aici_challenge/outputs/bathroom")
    parser.add_argument("--office-input", default="~/aici_challenge/outputs/office")
    parser.add_argument("--bathroom-detected", default="~/aici_challenge/outputs/bathroom_detected")
    parser.add_argument("--office-detected", default="~/aici_challenge/outputs/office_detected")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHALLENGE 1 - OCCUPANCY GRID MAP GENERATION")
    print("="*60)
    print("Features:")
    print("  - Detection clustering (removes duplicates)")
    print("  - Memory optimized (4x downsample, 5x frame skip)")
    print("  - Clean visualization with labels")
    print("="*60)
    
    if os.path.exists(os.path.expanduser(args.bathroom_input)):
        process_survey(args.bathroom_input, args.bathroom_detected, "bathroom")
    
    if os.path.exists(os.path.expanduser(args.office_input)):
        process_survey(args.office_input, args.office_detected, "office")
    
    print("\n" + "="*60)
    print(" CHALLENGE 1 COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()