"""
Use pre-trained diagram detection models from Roboflow.
No training needed - just inference on hosted models!
"""
from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def detect_diagrams_roboflow(
    image_path: str,
    api_key: str,
    model_choice: int = 1,
    confidence: float = 40,
    overlap: float = 30
) -> List[dict]:
    """
    Detect diagrams using pre-trained Roboflow models.
    
    Args:
        image_path: Path to the image file
        api_key: Your Roboflow API key (from https://app.roboflow.com/)
        model_choice: Which model to use:
            1 = Diagram Detection (IPCVCP) - General diagrams
            2 = Text and Diagram Finder - For educational content
            3 = Biology Paper Diagram Detection - For biology papers
        confidence: Confidence threshold (0-100)
        overlap: Overlap threshold for NMS (0-100)
    
    Returns:
        List of detections with bounding boxes and metadata
    """
    rf = Roboflow(api_key=api_key)
    
    # Model configurations
    models = {
        1: ("ipcvcp", "diagram-detection-wsnbk", 3),  # 923 images
        2: ("diagram-detection-set", "text-and-diagram-finder.v02", 3),  # 557 images
        3: ("aide-ai", "biology-paper-diagram-detection", 4)  # 876 images
    }
    
    workspace, project_name, version = models[model_choice]
    
    print(f"Using model: {project_name} (v{version})")
    print(f"Workspace: {workspace}")
    
    # Load the project and model
    project = rf.workspace(workspace).project(project_name)
    model = project.version(version).model
    
    # Run inference
    print(f"Running inference on: {image_path}")
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    
    # Extract predictions
    predictions = result.get('predictions', [])
    
    print(f"\nFound {len(predictions)} diagram(s)")
    
    detections = []
    for i, pred in enumerate(predictions):
        # Roboflow returns center coordinates + width/height
        x_center = pred['x']
        y_center = pred['y']
        width = pred['width']
        height = pred['height']
        
        # Convert to corner coordinates (x1, y1, x2, y2)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        detection = {
            'bbox': (x1, y1, x2, y2),
            'confidence': pred['confidence'],
            'class': pred['class'],
            'class_id': pred.get('class_id', 0)
        }
        
        detections.append(detection)
        
        print(f"  Diagram {i+1}: {pred['class']} (conf: {pred['confidence']:.2%})")
        print(f"    Box: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px")
    
    return detections


def visualize_roboflow_detections(
    image_path: str, 
    detections: List[dict], 
    output_path: Optional[str] = None
):
    """
    Visualize detections on the image.
    
    Args:
        image_path: Path to original image
        detections: List of detection dictionaries
        output_path: Where to save the annotated image
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class']
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label with background
        label = f"{class_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Detected {len(detections)} diagram(s)")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"\nSaved annotated image to: {output_path}")
    
    plt.show()


def extract_diagram_region(
    image_path: str, 
    bbox: Tuple[int, int, int, int], 
    output_path: Optional[str] = None,
    padding: int = 10
) -> Image.Image:
    """
    Extract diagram region from image.
    
    Args:
        image_path: Path to original image
        bbox: Bounding box as (x1, y1, x2, y2)
        output_path: Where to save the extracted diagram
        padding: Pixels to add around the diagram
    
    Returns:
        PIL Image of the extracted region
    """
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)
    
    # Crop
    diagram = img.crop((x1, y1, x2, y2))
    
    if output_path:
        diagram.save(output_path)
        print(f"Saved extracted diagram to: {output_path}")
    
    return diagram


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use pre-trained Roboflow models for diagram detection")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--image", type=str, default="page_9.png", help="Image path")
    parser.add_argument("--model", type=int, choices=[1, 2, 3], default=1,
                        help="Model: 1=General Diagrams, 2=Text&Diagram, 3=Biology Papers")
    parser.add_argument("--confidence", type=float, default=40, help="Confidence threshold (0-100)")
    parser.add_argument("--output", type=str, default="roboflow_result.jpg", help="Output path")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("\nERROR: API key required!")
        print("\nGet your free API key:")
        print("1. Go to https://app.roboflow.com/")
        print("2. Sign up or log in")
        print("3. Go to Settings → API")
        print("4. Copy your API key")
        print("\nThen run:")
        print(f'  python {__file__} --api-key YOUR_KEY --image page_9.png')
        exit(1)
    
    print("="*60)
    print("Roboflow Pre-trained Model Inference")
    print("="*60)
    
    # Detect diagrams
    detections = detect_diagrams_roboflow(
        image_path=args.image,
        api_key=args.api_key,
        model_choice=args.model,
        confidence=args.confidence
    )
    
    if detections:
        # Visualize
        visualize_roboflow_detections(args.image, detections, output_path=args.output)
        
        # Extract diagrams
        for i, det in enumerate(detections):
            extract_diagram_region(
                args.image, 
                det['bbox'], 
                output_path=f"roboflow_diagram_{i+1}.jpg"
            )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Annotated image: {args.output}")
        print(f"Extracted {len(detections)} diagram(s)")
        print("="*60)
    else:
        print("\nNo diagrams detected. Try:")
        print("  - Lowering confidence threshold: --confidence 20")
        print("  - Using a different model: --model 2 or --model 3")
