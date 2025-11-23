"""
YOLOv8/YOLOv11 Diagram Detection POC
Uses pre-trained YOLO models to detect objects/diagrams in question paper images.
Returns simple bounding boxes without normalization.
"""
import os
from typing import List, Tuple, Optional
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


def detect_diagrams_yolo(
    image_path: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    return_all: bool = False,
    debug: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Detect objects/diagrams in an image using YOLOv8/YOLOv11.
    
    Args:
        image_path: Path to the image file
        model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        conf_threshold: Confidence threshold for detections
        return_all: If True, return all detections. If False, return only the largest box.
        debug: If True, print debug information about all detections
    
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates
    """
    # Load model (will auto-download if not present)
    model = YOLO(model_name)
    
    # Run inference
    results = model(image_path, conf=conf_threshold, verbose=False)
    
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        if debug:
            print(f"\nTotal detections: {len(results[0].boxes)}")
        
        for box in results[0].boxes:
            # Get bounding box coordinates in xyxy format (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if debug:
                print(f"  Class: {model.names[cls]} | Conf: {conf:.2f} | Box: ({x1}, {y1}, {x2}, {y2})")
            
            boxes.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls,
                'class_name': model.names[cls]
            })
    
    # Sort by area (largest first)
    boxes.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
    
    if return_all:
        return [b['bbox'] for b in boxes]
    else:
        # Return largest box only
        return [boxes[0]['bbox']] if boxes else []


def visualize_detections(image_path: str, boxes: List[Tuple[int, int, int, int]], output_path: Optional[str] = None):
    """
    Draw bounding boxes on the image and save/display it.
    
    Args:
        image_path: Path to original image
        boxes: List of bounding boxes as (x1, y1, x2, y2)
        output_path: If provided, save the annotated image to this path
    """
    import matplotlib.pyplot as plt
    
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Box {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display
    plt.figure(figsize=(12, 8))
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
    
    Args:
        image_path: Path to original image
        bbox: Bounding box as (x1, y1, x2, y2)
        output_path: If provided, save the cropped diagram to this path
    
    Returns:
        PIL Image of the extracted region
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
    # Example usage
    print("YOLOv8 Diagram Detection POC")
    print("=" * 50)
    
    # Test with a sample image (you'll need to provide your own)
    test_image = "page_9.png"  # Replace with actual image path
    
    if os.path.exists(test_image):
        print(f"\nProcessing: {test_image}")
        
        # Detect diagrams
        print("\nDetecting diagrams with debug mode...")
        boxes = detect_diagrams_yolo(test_image, model_name="yolo11x.pt", conf_threshold=0.1, return_all=True, debug=True)
        
        if boxes:
            print(f"\nFound {len(boxes)} detection(s):")
            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                print(f"  Box {i+1}: ({x1}, {y1}, {x2}, {y2}) - Size: {width}x{height}px")
            
            # Visualize
            visualize_detections(test_image, boxes, output_path="output_annotated.jpg")
            
            # Extract largest diagram
            if boxes:
                extract_diagram_region(test_image, boxes[0], output_path="output_diagram.jpg")
        else:
            print("\nNo diagrams detected.")
    else:
        print(f"\nTest image not found: {test_image}")
        print("Please add a test image and update the path in the script.")
