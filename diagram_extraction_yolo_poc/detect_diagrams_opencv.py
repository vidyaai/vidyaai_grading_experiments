"""
OpenCV-based Diagram Detection
Uses contour detection to find diagrams in question paper images.
More suitable for diagrams than general object detection models.
"""
import os
from typing import List, Tuple, Optional
from PIL import Image
import cv2
import numpy as np


def detect_diagrams_opencv(
    image_path: str,
    min_area: int = 5000,
    max_area_ratio: float = 0.8,
    merge_nearby: bool = True,
    merge_distance: int = 100,
    debug: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Detect diagrams using OpenCV contour detection.
    
    Args:
        image_path: Path to the image file
        min_area: Minimum area in pixels for a valid diagram
        max_area_ratio: Maximum area as ratio of image size (to exclude full-page detections)
        merge_nearby: If True, merge nearby bounding boxes (for grouped diagrams)
        merge_distance: Maximum distance between boxes to merge them
        debug: If True, show intermediate processing steps
    
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w
    
    if debug:
        print(f"Image size: {w}x{h} ({img_area} pixels)")
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up and connect nearby elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Dilate to connect nearby diagram elements
    if merge_nearby:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_distance // 10, merge_distance // 10))
        morph = cv2.dilate(morph, dilate_kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"Found {len(contours)} contours")
    
    # Filter and extract bounding boxes
    boxes = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Filter by area
        area_ratio = area / img_area
        if area < min_area or area_ratio > max_area_ratio:
            if debug and area >= min_area:
                print(f"  Contour {i}: SKIPPED (too large - {area_ratio:.2%} of image)")
            continue
        
        # Calculate aspect ratio
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        if debug:
            print(f"  Contour {i}: Area={area}, Box=({x}, {y}, {x+w_box}, {y+h_box}), AR={aspect_ratio:.2f}")
        
        boxes.append({
            'bbox': (x, y, x + w_box, y + h_box),
            'area': area,
            'aspect_ratio': aspect_ratio
        })
    
    # Sort by area (largest first)
    boxes.sort(key=lambda x: x['area'], reverse=True)
    
    if debug:
        print(f"\nReturning {len(boxes)} valid diagram(s)")
    
    return [b['bbox'] for b in boxes]


def detect_diagrams_by_density(
    image_path: str,
    tile_size: int = 100,
    density_threshold: float = 0.3,
    min_area: int = 5000,
    debug: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Detect diagram regions by analyzing pixel density in tiles.
    Works well for graphs and diagrams with dense line work.
    
    Args:
        image_path: Path to the image file
        tile_size: Size of tiles to analyze
        density_threshold: Minimum density of non-white pixels to consider a diagram
        min_area: Minimum area for valid diagrams
        debug: If True, print debug information
    
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates
    """
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create density map
    density_map = np.zeros((h // tile_size + 1, w // tile_size + 1), dtype=np.float32)
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = binary[i:min(i+tile_size, h), j:min(j+tile_size, w)]
            density = np.sum(tile > 0) / (tile.shape[0] * tile.shape[1])
            density_map[i // tile_size, j // tile_size] = density
    
    # Find regions with high density
    high_density = (density_map > density_threshold).astype(np.uint8) * 255
    
    # Dilate to connect nearby regions
    kernel = np.ones((3, 3), np.uint8)
    high_density = cv2.dilate(high_density, kernel, iterations=2)
    
    # Find contours in density map
    contours, _ = cv2.findContours(high_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        # Scale back to original image coordinates
        x, y, w_box, h_box = cv2.boundingRect(contour)
        x1 = x * tile_size
        y1 = y * tile_size
        x2 = min((x + w_box) * tile_size, w)
        y2 = min((y + h_box) * tile_size, h)
        
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            boxes.append((x1, y1, x2, y2))
            if debug:
                print(f"  Density region: ({x1}, {y1}, {x2}, {y2}) - Area: {area}")
    
    return boxes


def visualize_detections(image_path: str, boxes: List[Tuple[int, int, int, int]], output_path: Optional[str] = None):
    """
    Draw bounding boxes on the image and save/display it.
    """
    import matplotlib.pyplot as plt
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Diagram {i+1}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Detected {len(boxes)} diagram(s)")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved annotated image to: {output_path}")
    
    plt.show()


def extract_diagram_region(image_path: str, bbox: Tuple[int, int, int, int], output_path: Optional[str] = None) -> Image.Image:
    """
    Extract and save the diagram region from the image.
    """
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    
    # Crop the region
    diagram = img.crop((x1, y1, x2, y2))
    
    if output_path:
        diagram.save(output_path)
        print(f"Saved extracted diagram to: {output_path}")
    
    return diagram


if __name__ == "__main__":
    print("OpenCV Diagram Detection POC")
    print("=" * 50)
    
    test_image = "page_9.png"
    
    if os.path.exists(test_image):
        print(f"\nProcessing: {test_image}\n")
        
        # Method 1: Contour detection with merging
        print("Method 1: Contour Detection (with merging)")
        print("-" * 50)
        boxes_contour = detect_diagrams_opencv(test_image, min_area=10000, merge_nearby=True, merge_distance=150, debug=True)
        
        # Method 2: Density-based detection
        print("\nMethod 2: Density-based Detection")
        print("-" * 50)
        boxes_density = detect_diagrams_by_density(test_image, debug=True)
        
        # Use the method that found more diagrams
        if len(boxes_contour) > 0 or len(boxes_density) > 0:
            boxes = boxes_contour if len(boxes_contour) >= len(boxes_density) else boxes_density
            method = "Contour" if len(boxes_contour) >= len(boxes_density) else "Density"
            
            print(f"\n{'='*50}")
            print(f"Using {method} method results: {len(boxes)} diagram(s)")
            print(f"{'='*50}")
            
            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                print(f"  Diagram {i+1}: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px")
            
            # Visualize
            visualize_detections(test_image, boxes, output_path="opencv_output_annotated.jpg")
            
            # Extract diagrams
            for i, bbox in enumerate(boxes):
                extract_diagram_region(test_image, bbox, output_path=f"opencv_diagram_{i+1}.jpg")
        else:
            print("\nNo diagrams detected with either method.")
    else:
        print(f"\nTest image not found: {test_image}")
