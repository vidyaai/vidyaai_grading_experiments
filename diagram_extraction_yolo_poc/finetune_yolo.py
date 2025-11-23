"""
Fine-tune YOLO models on diagram detection datasets from Roboflow.

This script downloads pre-labeled diagram datasets and trains a YOLOv11 model
specifically for detecting diagrams in question papers, including axis labels
and annotations that OpenCV methods miss.
"""
import os
from ultralytics import YOLO


def download_dataset_from_roboflow(api_key: str, dataset_choice: int = 1):
    """
    Download diagram detection dataset from Roboflow.
    
    Args:
        api_key: Your Roboflow API key (get from https://app.roboflow.com/)
        dataset_choice: Which dataset to use
            1 = Biology Paper Diagram Detection (876 images) - RECOMMENDED
            2 = Text and Diagram Finder (557 images)
            3 = Diagram Detection (923 images)
    
    Returns:
        Path to the downloaded dataset
    """
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    
    datasets = {
        1: ("aide-ai", "biology-paper-diagram-detection", 4),
        2: ("diagram-detection-set", "text-and-diagram-finder.v02", 3),
        3: ("ipcvcp", "diagram-detection-wsnbk", 3)
    }
    
    workspace, project_name, version = datasets[dataset_choice]
    
    print(f"\nDownloading dataset: {project_name}")
    print(f"Workspace: {workspace}, Version: {version}")
    
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov11")
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_yolo(
    dataset_path: str,
    model_size: str = "yolo11n.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0"
):
    """
    Fine-tune YOLO model on diagram detection dataset.
    
    Args:
        dataset_path: Path to dataset.yaml file
        model_size: Base model to start from (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size (reduce if GPU memory issues)
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
    """
    print(f"\n{'='*60}")
    print(f"Starting YOLO Fine-tuning")
    print(f"{'='*60}")
    print(f"Model: {model_size}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load pre-trained model
    model = YOLO(model_size)
    
    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")
    print(f"Results saved to: runs/detect/train/")
    
    return results


def validate_model(model_path: str, dataset_path: str):
    """
    Validate the trained model on the test set.
    
    Args:
        model_path: Path to the trained model (best.pt)
        dataset_path: Path to dataset.yaml file
    """
    print(f"\nValidating model: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=dataset_path)
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


def test_on_image(model_path: str, image_path: str, save_path: str = None):
    """
    Test the trained model on a single image.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to test image
        save_path: Where to save the annotated result
    """
    model = YOLO(model_path)
    
    results = model(image_path)
    
    # Print detections
    for result in results:
        boxes = result.boxes
        print(f"\nFound {len(boxes)} diagram(s):")
        for i, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            x1, y1, x2, y2 = map(int, coords)
            print(f"  Diagram {i+1}: {class_name} (conf: {conf:.2f})")
            print(f"    Box: ({x1}, {y1}, {x2}, {y2}) - Size: {x2-x1}x{y2-y1}px")
    
    # Save annotated image
    if save_path:
        annotated = results[0].plot()
        import cv2
        cv2.imwrite(save_path, annotated)
        print(f"\nSaved annotated image to: {save_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for diagram detection")
    parser.add_argument("--api-key", type=str, help="Roboflow API key", default=None)
    parser.add_argument("--dataset", type=int, choices=[1, 2, 3], default=1,
                        help="Dataset choice: 1=Biology Papers, 2=Text&Diagram, 3=General Diagrams")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base model size (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset.yaml if already downloaded")
    parser.add_argument("--test-image", type=str, help="Test image path after training")
    
    args = parser.parse_args()
    
    # Download dataset
    if not args.skip_download:
        if not args.api_key:
            print("\nNO API KEY PROVIDED")
            print("="*60)
            print("To download datasets automatically, you need a Roboflow API key.")
            print("\nGet your API key:")
            print("1. Go to https://app.roboflow.com/")
            print("2. Sign up or log in")
            print("3. Go to Settings → API")
            print("4. Copy your API key")
            print("\nThen run:")
            print(f'  python {__file__} --api-key YOUR_API_KEY')
            print("\nOR download manually:")
            print("1. Visit the dataset URL")
            print("2. Click 'Download Dataset'")
            print("3. Select 'YOLOv11' format")
            print("4. Extract to 'datasets/' folder")
            print("5. Run with: --skip-download --dataset-path datasets/data.yaml")
            print("="*60)
            exit(1)
        
        dataset_path = download_dataset_from_roboflow(args.api_key, args.dataset)
        dataset_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        dataset_yaml = args.dataset_path
        if not dataset_yaml or not os.path.exists(dataset_yaml):
            print(f"ERROR: Dataset path not found: {dataset_yaml}")
            exit(1)
    
    # Train the model
    print("\n" + "="*60)
    print("STEP 1: Training Model")
    print("="*60)
    train_yolo(
        dataset_path=dataset_yaml,
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=args.device
    )
    
    # Validate
    print("\n" + "="*60)
    print("STEP 2: Validating Model")
    print("="*60)
    best_model = "runs/detect/train/weights/best.pt"
    validate_model(best_model, dataset_yaml)
    
    # Test on sample image if provided
    if args.test_image and os.path.exists(args.test_image):
        print("\n" + "="*60)
        print("STEP 3: Testing on Sample Image")
        print("="*60)
        test_on_image(best_model, args.test_image, save_path="test_result_annotated.jpg")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Trained model: {best_model}")
    print(f"\nTo use the model:")
    print(f'  python detect_diagrams.py --model {best_model} --image your_image.jpg')
    print("="*60 + "\n")
