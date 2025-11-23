"""
Quick training script for the downloaded diagram detection dataset.
Since pre-trained weights aren't available for free download, we'll do a quick fine-tune.
"""
from ultralytics import YOLO
import os


def quick_train_diagram_detector(
    dataset_path: str = "pretrained_models/diagram_detection_v1/data.yaml",
    epochs: int = 30,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = None
):
    """
    Quick training on the diagram detection dataset.
    
    Args:
        dataset_path: Path to data.yaml
        epochs: Number of epochs (30 is enough for good results)
        img_size: Image size
        batch_size: Batch size (reduce if GPU memory issues)
        device: Device to use ('0' for GPU, 'cpu' for CPU, None for auto)
    """
    import torch
    
    # Auto-detect device if not specified
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("Quick Training - Diagram Detection Model")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*70)
    
    # Start with nano model for speed
    model = YOLO("yolo11n.pt")
    
    # Train
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=10,
        save=True,
        plots=True,
        device=device,
        verbose=True,
        name="diagram_detector"
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Model saved to: runs/detect/diagram_detector/weights/best.pt")
    print("\nNow you can use it:")
    print("  python use_free_model.py --model-path runs/detect/diagram_detector/weights/best.pt --image page_9.png")
    print("="*70)
    
    return "runs/detect/diagram_detector/weights/best.pt"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick train diagram detector")
    parser.add_argument("--dataset", type=str, 
                        default="pretrained_models/diagram_detection_v1/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device: 0 for GPU, cpu for CPU, None for auto")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("\nFirst download the dataset:")
        print("  python use_free_model.py --download --api-key YOUR_KEY --model-choice 1")
        exit(1)
    
    quick_train_diagram_detector(
        dataset_path=args.dataset,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=args.device
    )
