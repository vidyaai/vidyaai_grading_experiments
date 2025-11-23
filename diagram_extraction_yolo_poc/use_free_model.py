"""
Download and use pre-trained YOLO models from Roboflow Universe for FREE local inference.
No API calls needed after download - runs completely offline.
"""
import os
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def download_pretrained_model(
    api_key: str,
    model_choice: int = 1,
    download_dir: str = "pretrained_models"
) -> str:
    """
    Download pre-trained model weights from Roboflow (ONE-TIME, then free forever).
    
    Args:
        api_key: Roboflow API key (only needed for initial download)
        model_choice: Which model to download:
            1 = Diagram Detection (IPCVCP) - 923 images
            2 = Text and Diagram Finder - 557 images  
        download_dir: Where to save the model
    
    Returns:
        Path to the downloaded model weights
    """
    os.makedirs(download_dir, exist_ok=True)
    
    rf = Roboflow(api_key=api_key)
    
    models = {
        1: ("ipcvcp", "diagram-detection-wsnbk", 1, "diagram_detection"),
        2: ("diagram-detection-set", "text-and-diagram-finder.v02", 3, "text_diagram_finder"),
    }
    
    workspace, project_name, version, model_name = models[model_choice]
    model_path = os.path.join(download_dir, f"{model_name}_v{version}")
    
    # Check if already downloaded
    weights_file = os.path.join(model_path, "weights", "best.pt")
    if os.path.exists(weights_file):
        print(f"Model already downloaded: {weights_file}")
        return weights_file
    
    print(f"Downloading model: {project_name} (v{version})")
    print(f"This is a ONE-TIME download. Model will be saved for offline use.")
    
    # Download dataset with model weights
    project = rf.workspace(workspace).project(project_name)
    
    # Try to get available versions if the specified one fails
    try:
        dataset = project.version(version).download("yolov11", location=model_path)
    except RuntimeError as e:
        print(f"\nVersion {version} not found. Checking available versions...")
        # Try to list and use the latest version
        try:
            # Get project info to see available versions
            print(f"Available versions: Check https://universe.roboflow.com/{workspace}/{project_name}")
            print(f"Trying version 1...")
            dataset = project.version(1).download("yolov11", location=model_path)
        except Exception as e2:
            print(f"Error: {e2}")
            raise RuntimeError(f"Could not download model. Please check the version at https://universe.roboflow.com/{workspace}/{project_name}")
    
    # Find the weights file
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".pt"):
                weights_file = os.path.join(root, file)
                print(f"\nModel downloaded successfully!")
                print(f"Weights: {weights_file}")
                return weights_file
    
    raise FileNotFoundError("Model weights not found after download")


def detect_diagrams_local(
    image_path: str,
    model_path: str,
    confidence: float = 0.25,
    verbose: bool = True
) -> List[dict]:
    """
    Detect diagrams using local YOLO model (100% FREE, runs offline).
    
    Args:
        image_path: Path to the image
        model_path: Path to model weights (.pt file)
        confidence: Confidence threshold (0.0 - 1.0)
        verbose: Print detection info
    
    Returns:
        List of detections with bounding boxes
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {image_path}")
    results = model(image_path, conf=confidence, verbose=False)
    
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        if verbose:
            print(f"\nFound {len(boxes)} diagram(s):")
        
        for i, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': class_name,
                'class_id': cls
            }
            
            detections.append(detection)
            
            if verbose:
                print(f"  Diagram {i+1}: {class_name} (conf: {conf:.2%})")
                print(f"    Box: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px")
    
    return detections


def visualize_detections(
    image_path: str, 
    detections: List[dict], 
    output_path: Optional[str] = None
):
    """Visualize detections on the image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class']
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Draw label
        label = f"{class_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 5, y1 - 8), font, font_scale, (0, 0, 0), thickness)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Detected {len(detections)} diagram(s)")
    
    if output_path:
        plt.savefig(f"{output_path}/free_model_result.jpg", bbox_inches='tight', dpi=150)
        print(f"\nSaved annotated image to: {output_path}")
    
    plt.show()


def extract_diagram_region(
    image_path: str, 
    bbox: Tuple[int, int, int, int], 
    output_path: Optional[str] = None,
    padding: int = 20
) -> Image.Image:
    """Extract diagram region from image with padding."""
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    
    # Add padding
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Use pre-trained models locally for FREE")
    parser.add_argument("--download", action="store_true", help="Download model (one-time only)")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (only for initial download)")
    parser.add_argument("--model-choice", type=int, choices=[1, 2], default=1,
                        help="Model: 1=Diagram Detection, 2=Text&Diagram Finder")
    parser.add_argument("--model-path", type=str, help="Path to downloaded model weights (.pt)")
    parser.add_argument("--image", type=str, default="page_9.png", help="Image to process")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--output", type=str, default="free_model_result.jpg", help="Output path")
    
    args = parser.parse_args()
    
    print("="*70)
    print("FREE Local Diagram Detection (Pre-trained Model)")
    print("="*70)
    
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # Download mode
    if args.download:
        if not args.api_key:
            print("\nTo download the model, you need a Roboflow API key (free account):")
            print("1. Go to https://app.roboflow.com/")
            print("2. Sign up (free)")
            print("3. Go to Settings → API")
            print("4. Copy your API key")
            print("\nThen run:")
            print(f"  python {__file__} --download --api-key YOUR_KEY --model-choice 1")
            print("\nAfter download, you can use the model offline WITHOUT API key!")
            exit(1)
        
        model_path = download_pretrained_model(args.api_key, args.model_choice)
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"Model saved to: {model_path}")
        print("\nNow you can use it offline without API key:")
        print(f"  python {__file__} --model-path {model_path} --image page_9.png")
        print("="*70)
        exit(0)
    
    # Inference mode
    if not args.model_path:
        print("\nERROR: Model path required!")
        print("\nFirst-time setup:")
        print("  1. Download model (one-time, requires free Roboflow account):")
        print(f"     python {__file__} --download --api-key YOUR_KEY")
        print("\n  2. Then use it forever for free:")
        print(f"     python {__file__} --model-path pretrained_models/.../best.pt --image page_9.png")
        print("\nOR manually download from:")
        print("  https://universe.roboflow.com/ipcvcp/diagram-detection-wsnbk")
        print("  Click 'Download Dataset' → 'YOLOv11' format → Export as .pt weights")
        exit(1)
    
    # Run inference
    detections = detect_diagrams_local(
        image_path=args.image,
        model_path=args.model_path,
        confidence=args.confidence
    )
    
    if detections:
        # Visualize
        visualize_detections(args.image, detections, output_path=args.output)
        
        # Extract diagrams
        for i, det in enumerate(detections):
            extract_diagram_region(
                args.image, 
                det['bbox'], 
                output_path=f"{args.output}/free_model_diagram_{i+1}.jpg"
            )
        
        print("\n" + "="*70)
        print("SUCCESS! (100% FREE - No API calls used)")
        print("="*70)
        print(f"Annotated image: {args.output}")
        print(f"Extracted {len(detections)} diagram(s)")
        print("="*70)
    else:
        print("\nNo diagrams detected. Try:")
        print("  - Lowering confidence: --confidence 0.15")
        print("  - Different model: --model-choice 2")
