#!/usr/bin/env python3

#Challenge 3: Generate Aligned Occupancy Grid Maps : Takes the provided  files and aligns them using computed transform
import cv2
import numpy as np
import yaml
import json
from pathlib import Path

def load_pgm_map(pgm_path):
    """Load PGM map file"""
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load map: {pgm_path}")
    return img

def load_yaml_config(yaml_path):
    """Load map YAML configuration"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_alignment_transform(bathroom_config, office_config):
   
    # Get origins
    bathroom_origin = np.array(bathroom_config['origin'][:2])
    office_origin = np.array(office_config['origin'][:2])
    
    # Get resolutions
    bathroom_res = bathroom_config['resolution']
    office_res = office_config['resolution']
    
    # Compute translation in meters
    translation_meters = office_origin - bathroom_origin
    
    # Convert to pixels (using bathroom resolution as reference)
    translation_pixels = translation_meters / bathroom_res
    
    print(f"Bathroom origin: {bathroom_origin}")
    print(f"Office origin: {office_origin}")
    print(f"Translation (meters): {translation_meters}")
    print(f"Translation (pixels): {translation_pixels}")
    
    return translation_pixels, translation_meters

def create_aligned_visualization(bathroom_map, office_map, translation_pixels):
    
    print(f"\nCreating aligned visualization...")
    print(f"Bathroom map shape: {bathroom_map.shape}")
    print(f"Office map shape: {office_map.shape}")
    print(f"Translation: {translation_pixels}")
    
    # Calculate canvas size needed to fit both maps
    tx, ty = int(translation_pixels[0]), int(translation_pixels[1])
    
    # Calculate bounding box for both maps
    # Bathroom at origin (0,0)
    bathroom_bounds = {
        'min_x': 0,
        'max_x': bathroom_map.shape[1],
        'min_y': 0,
        'max_y': bathroom_map.shape[0]
    }
    
    # Office at translated position
    office_bounds = {
        'min_x': tx,
        'max_x': tx + office_map.shape[1],
        'min_y': ty,
        'max_y': ty + office_map.shape[0]
    }
    
    # Overall bounding box
    overall_min_x = min(bathroom_bounds['min_x'], office_bounds['min_x'])
    overall_max_x = max(bathroom_bounds['max_x'], office_bounds['max_x'])
    overall_min_y = min(bathroom_bounds['min_y'], office_bounds['min_y'])
    overall_max_y = max(bathroom_bounds['max_y'], office_bounds['max_y'])
    
    # Canvas dimensions with padding
    padding = 100
    canvas_width = int(overall_max_x - overall_min_x) + 2 * padding
    canvas_height = int(overall_max_y - overall_min_y) + 2 * padding
    
    print(f"Canvas size: {canvas_width}x{canvas_height}")
    print(f"Overall bounds: X=[{overall_min_x}, {overall_max_x}], Y=[{overall_min_y}, {overall_max_y}]")
    
    # Create color canvas (white background)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Calculate offsets to center everything
    bathroom_offset_x = padding - int(overall_min_x)
    bathroom_offset_y = padding - int(overall_min_y)
    
    # Position office map based on translation
    office_offset_x = bathroom_offset_x + tx
    office_offset_y = bathroom_offset_y + ty
    
    print(f"Bathroom position: ({bathroom_offset_x}, {bathroom_offset_y})")
    print(f"Office position: ({office_offset_x}, {office_offset_y})")
    
    # Create colored versions of maps
    # Bathroom = Green channel
    bathroom_colored = np.zeros((bathroom_map.shape[0], bathroom_map.shape[1], 3), dtype=np.uint8)
    bathroom_colored[:, :, 1] = bathroom_map  # Green channel
    
    # Office = Blue channel
    office_colored = np.zeros((office_map.shape[0], office_map.shape[1], 3), dtype=np.uint8)
    office_colored[:, :, 0] = office_map  # Blue channel
    
    # Create masks for occupied space (black pixels = obstacles)
    bathroom_mask = bathroom_map < 250
    office_mask = office_map < 250
    
    # Place bathroom map (green tint)
    y1_b = bathroom_offset_y
    y2_b = bathroom_offset_y + bathroom_map.shape[0]
    x1_b = bathroom_offset_x
    x2_b = bathroom_offset_x + bathroom_map.shape[1]
    
    # Ensure within bounds
    if y1_b >= 0 and x1_b >= 0 and y2_b <= canvas_height and x2_b <= canvas_width:
        # Create bathroom mask for non-gray areas
        bathroom_free = bathroom_map > 250  # Free space (white)
        bathroom_occupied = bathroom_map < 100  # Obstacles (black)
        
        # Apply bathroom with green tint (BGR format)
        canvas[y1_b:y2_b, x1_b:x2_b][bathroom_free] = [220, 255, 220]  # Light green (BGR)
        canvas[y1_b:y2_b, x1_b:x2_b][bathroom_occupied] = [0, 100, 0]  # Dark green (BGR)
        print(f" Bathroom map placed")
    else:
        print(f"Bathroom map out of bounds!")
    
    # Place office map (blue tint)  
    y1_o = office_offset_y
    y2_o = office_offset_y + office_map.shape[0]
    x1_o = office_offset_x
    x2_o = office_offset_x + office_map.shape[1]
    
    print(f"Office bounds: Y=[{y1_o}, {y2_o}], X=[{x1_o}, {x2_o}]")
    print(f"Canvas bounds: Y=[0, {canvas_height}], X=[0, {canvas_width}]")
    
    # Ensure within bounds
    if y1_o >= 0 and x1_o >= 0 and y2_o <= canvas_height and x2_o <= canvas_width:
        # Create office mask for non-gray areas
        office_free = office_map > 250  # Free space (white)
        office_occupied = office_map < 100  # Obstacles (black)
        
        # Apply office with blue tint (overlay on existing)
        office_region = canvas[y1_o:y2_o, x1_o:x2_o].copy()
        
        # Where office has free space, apply blue (BGR format!)
        office_region[office_free] = [255, 220, 220]  # Light blue for free space (BGR)
        # Where office has obstacles, apply dark blue (BGR format!)
        office_region[office_occupied] = [100, 0, 0]  # Dark blue for obstacles (BGR)
        
        canvas[y1_o:y2_o, x1_o:x2_o] = office_region
        print(f"Office map placed")
        
        # Check if there's overlap for logging purposes only
        overlap_y_start = max(y1_b, y1_o)
        overlap_y_end = min(y2_b, y2_o)
        overlap_x_start = max(x1_b, x1_o)
        overlap_x_end = min(x2_b, x2_o)
        
        if overlap_y_start < overlap_y_end and overlap_x_start < overlap_x_end:
            overlap_area = (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)
            print(f"Maps overlap in region Y=[{overlap_y_start}, {overlap_y_end}], X=[{overlap_x_start}, {overlap_x_end}]")
            print(f"Overlap area: {overlap_area} pixels")
        else:
            print(f"No spatial overlap between maps")
    else:
        print(f"Office map out of bounds!")
        # If office is out of bounds, we need a bigger canvas
        return None  # Signal to retry with bigger canvas
    
    # Add simple title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Aligned Maps", (20, 40), font, 1.2, (0, 0, 0), 2)
    
    # Legend
    legend_y = canvas_height - 140
    legend_x = 20
    
    # Bathroom (dark green) - BGR format
    cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 40, legend_y + 30), (0, 100, 0), -1)
    cv2.putText(canvas, "Bathroom", (legend_x + 50, legend_y + 22), font, 0.7, (0, 0, 0), 2)
    
    # Office (dark blue) - BGR format  
    cv2.rectangle(canvas, (legend_x, legend_y + 40), (legend_x + 40, legend_y + 70), (100, 0, 0), -1)
    cv2.putText(canvas, "Office", (legend_x + 50, legend_y + 62), font, 0.7, (0, 0, 0), 2)
    
    return canvas

def save_transform(translation_pixels, translation_meters, bathroom_config, office_config, output_path):
    #Save alignment transform to JSON
    
    # Create transformation matrix (2D affine: translation only, no rotation)
    # [1  0  tx]
    # [0  1  ty]
    transform_matrix = [
        [1.0, 0.0, float(translation_pixels[0])],
        [0.0, 1.0, float(translation_pixels[1])]
    ]
    
    transform_data = {
        "environment": "bathroom_and_office",
        "description": "Transform to align office map with bathroom map",
        "transformation_matrix": transform_matrix,
        "translation_pixels": [float(translation_pixels[0]), float(translation_pixels[1])],
        "translation_meters": [float(translation_meters[0]), float(translation_meters[1])],
        "resolution": bathroom_config['resolution'],
        "bathroom_origin": bathroom_config['origin'][:2],
        "office_origin": office_config['origin'][:2]
    }
    
    # Save JSON
    json_path = output_path.parent / "bathroom_office_alignment_transform.json"
    with open(json_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    print(f"\n Saved transform: {json_path}")
    
    # Save YAML
    yaml_path = output_path.parent / "bathroom_office_alignment_transform.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(transform_data, f, default_flow_style=False)
    print(f"Saved transform: {yaml_path}")
    
    return transform_data

def main():
    print("="*70)
    print("Challenge 3: Aligned Occupancy Grid Maps Generator")
    print("="*70)
    
    # Define paths
    base = Path("~/aici_challenge").expanduser()
    gridmap_path = base / "gridmap"
    output_path = base / "outputs" / "challenge3"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load bathroom map
    print("\nLoading bathroom map...")
    bathroom_pgm = gridmap_path / "bathroom" / "bathroom.pgm"
    bathroom_yaml = gridmap_path / "bathroom" / "bathroom.yaml"
    
    # Try alternative names if not found
    if not bathroom_yaml.exists():
        bathroom_yaml = gridmap_path / "bathroom" / "room.yaml"
    
    bathroom_map = load_pgm_map(bathroom_pgm)
    bathroom_config = load_yaml_config(bathroom_yaml)
    print(f" Bathroom map loaded: {bathroom_map.shape}")
    print(f"  Origin: {bathroom_config['origin'][:2]}")
    print(f"  Resolution: {bathroom_config['resolution']} m/pixel")
    
    # Load office map
    print("\nLoading office map...")
    office_pgm = gridmap_path / "office" / "officeroom.pgm"
    office_yaml = gridmap_path / "office" / "officeroom.yaml"
    
    # Try alternative names if not found
    if not office_yaml.exists():
        office_yaml = gridmap_path / "office" / "room.yaml"
    
    office_map = load_pgm_map(office_pgm)
    office_config = load_yaml_config(office_yaml)
    print(f"Office map loaded: {office_map.shape}")
    print(f"  Origin: {office_config['origin'][:2]}")
    print(f"  Resolution: {office_config['resolution']} m/pixel")
    
    # Compute alignment transform
    print("\n" + "="*70)
    print("Computing alignment transform:")
    print("="*70)
    translation_pixels, translation_meters = compute_alignment_transform(
        bathroom_config, office_config
    )
    
    # Create aligned visualization
    print("\n" + "="*70)
    print("Creating aligned visualization:")
    print("="*70)
    aligned_canvas = create_aligned_visualization(
        bathroom_map, office_map, translation_pixels
    )
    
    # Save outputs
    print("\n" + "="*70)
    print("Saving outputs:")
    print("="*70)
    
    output_image = output_path / "aligned_occupancy_maps.png"
    cv2.imwrite(str(output_image), aligned_canvas)
    print(f"Saved aligned map: {output_image}")
    
    # Save transform
    transform_data = save_transform(
        translation_pixels, translation_meters,
        bathroom_config, office_config, output_path
    )
    
    print("\n" + "="*70)
    print("Challenge 3 completed")
    print("="*70)
    print(f"\nTransform summary:")
    print(f"Translation: {translation_meters[0]:.2f}m, {translation_meters[1]:.2f}m")
    print(f"Pixels: {translation_pixels[0]:.1f}px, {translation_pixels[1]:.1f}px")

if __name__ == "__main__":
    main()