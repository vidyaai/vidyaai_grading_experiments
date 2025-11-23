"""
Grouped Diagram Detection
Detects individual diagram elements and merges them into logical groups.
"""
import os
from typing import List, Tuple, Optional
from PIL import Image
import cv2
import numpy as np


def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], horizontal_threshold: int = 200, vertical_threshold: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Merge bounding boxes that are close to each other.
    
    Args:
        boxes: List of bounding boxes as (x1, y1, x2, y2)
        horizontal_threshold: Maximum horizontal distance to merge boxes
        vertical_threshold: Maximum vertical distance to merge boxes
    
    Returns:
        List of merged bounding boxes
    """
    if not boxes:
        return []
    
    # Convert to list of lists for easier manipulation
    merged = [[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes]
    
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                x1a, y1a, x2a, y2a = merged[i]
                x1b, y1b, x2b, y2b = merged[j]
                
                # Check if boxes are close enough to merge
                # Horizontal distance
                h_dist = min(abs(x2a - x1b), abs(x2b - x1a))
                if x1b > x2a:
                    h_dist = x1b - x2a
                elif x1a > x2b:
                    h_dist = x1a - x2b
                else:
                    h_dist = 0  # Overlapping
                
                # Vertical distance
                v_dist = min(abs(y2a - y1b), abs(y2b - y1a))
                if y1b > y2a:
                    v_dist = y1b - y2a
                elif y1a > y2b:
                    v_dist = y1a - y2b
                else:
                    v_dist = 0  # Overlapping
                
                # Check if they're on similar vertical or horizontal level
                v_overlap = not (y2a < y1b or y2b < y1a)
                h_overlap = not (x2a < x1b or x2b < x1a)
                
                should_merge = False
                if v_overlap and h_dist <= horizontal_threshold:
                    # Same row, close horizontally
                    should_merge = True
                elif h_overlap and v_dist <= vertical_threshold:
                    # Same column, close vertically
                    should_merge = True
                elif h_dist <= horizontal_threshold and v_dist <= vertical_threshold:
                    # Close both ways
                    should_merge = True
                
                if should_merge:
                    # Merge boxes
                    merged[i] = [
                        min(x1a, x1b),
                        min(y1a, y1b),
                        max(x2a, x2b),
                        max(y2a, y2b)
                    ]
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    
    return [tuple(box) for box in merged]


def detect_diagrams_grouped(
    image_path: str,
    min_area: int = 5000,
    max_area_ratio: float = 0.5,
    horizontal_merge_threshold: int = 200,
    vertical_merge_threshold: int = 100,
    debug: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Detect diagrams and group nearby ones together.
    
    Args:
        image_path: Path to the image file
        min_area: Minimum area in pixels for a valid diagram component
        max_area_ratio: Maximum area as ratio of image size
        horizontal_merge_threshold: Max horizontal distance to merge boxes
        vertical_merge_threshold: Max vertical distance to merge boxes
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
    
    # Morphological operations to clean up (minimal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"Found {len(contours)} contours")
    
    # Filter and extract bounding boxes
    initial_boxes = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Filter by area
        area_ratio = area / img_area
        if area < min_area or area_ratio > max_area_ratio:
            continue
        
        initial_boxes.append((x, y, x + w_box, y + h_box))
    
    if debug:
        print(f"Found {len(initial_boxes)} initial diagram components")
    
    # Merge nearby boxes
    merged_boxes = merge_nearby_boxes(
        initial_boxes, 
        horizontal_threshold=horizontal_merge_threshold,
        vertical_threshold=vertical_merge_threshold
    )
    
    if debug:
        print(f"After merging: {len(merged_boxes)} diagram group(s)")
        for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
            area = (x2 - x1) * (y2 - y1)
            print(f"  Group {i+1}: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px, Area: {area}")
    
    # Sort by area (largest first)
    merged_boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    
    return merged_boxes


def visualize_detections(image_path: str, boxes: List[Tuple[int, int, int, int]], output_path: Optional[str] = None):
    """Draw bounding boxes on the image."""
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        label = f"Diagram {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Detected {len(boxes)} diagram group(s)")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved annotated image to: {output_path}")
    
    plt.show()


def extract_diagram_region(image_path: str, bbox: Tuple[int, int, int, int], output_path: Optional[str] = None) -> Image.Image:
    """Extract and save the diagram region."""
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    
    # Add small padding
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)
    
    diagram = img.crop((x1, y1, x2, y2))
    
    if output_path:
        diagram.save(output_path)
        print(f"Saved extracted diagram to: {output_path}")
    
    return diagram


if __name__ == "__main__":
    print("Grouped Diagram Detection POC")
    print("=" * 50)
    
    test_image = "page_9.png"
    
    if os.path.exists(test_image):
        print(f"\nProcessing: {test_image}\n")
        
        # Detect and group diagrams
        boxes = detect_diagrams_grouped(
            test_image, 
            min_area=5000,  # Lower threshold to catch individual graphs
            horizontal_merge_threshold=700,  # Distance to merge horizontally (merge graphs in same row)
            vertical_merge_threshold=300,    # Distance to merge vertically (merge different rows)
            debug=True
        )
        
        if boxes:
            print(f"\n{'='*50}")
            print(f"Final result: {len(boxes)} diagram group(s)")
            print(f"{'='*50}")
            
            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                print(f"  Diagram {i+1}: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px")
            
            # Visualize
            visualize_detections(test_image, boxes, output_path="grouped_output_annotated.jpg")
            
            # Extract diagrams
            for i, bbox in enumerate(boxes):
                extract_diagram_region(test_image, bbox, output_path=f"grouped_diagram_{i+1}.jpg")
        else:
            print("\nNo diagrams detected.")
    else:
        print(f"\nTest image not found: {test_image}")
