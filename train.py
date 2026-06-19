from ultralytics import YOLO
import os

def main():
    print("Loading pretrained YOLOv26n model...")
    model = YOLO('yolo26n.pt')
    
    print("Starting YOLOv26 model training on plant stress dataset...")
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=3,
        imgsz=640,
        batch=8,
        workers=0,
        device='cpu'  # Force CPU since GPU is not active
    )
    print("-------------------------------------------------")
    print("YOLOv26 training completed successfully!")
    print("Training runs, weights, and metrics saved to: runs/detect/train")
    print("-------------------------------------------------")

if __name__ == '__main__':
    main()
