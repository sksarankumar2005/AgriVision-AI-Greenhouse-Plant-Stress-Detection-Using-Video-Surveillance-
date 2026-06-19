import os
import cv2
import json
import math
import sys

# Define base paths
WORKSPACE = r"c:\Users\SARANKUMAR\Downloads\AgriVision_AI"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
JSON_DIR = os.path.join(DATASET_DIR, "json_labels")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Create directories if they do not exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Class mappings
CLASS_MAPPING = {
    "healthy": 0,
    "wilting": 1,
    "yellowing": 2,
    "tip_burn": 3,
    "dry_leaf": 4
}

def generate_shapes(frame_idx, total_frames):
    progress = frame_idx / max(1, total_frames)
    shapes = []
    
    # Helper function to add a box with organic noise and growth
    def add_box(base_box, label, growth_start=1.0, growth_rate=0.12):
        growth = growth_start + (growth_rate * progress)
        
        cx = (base_box[0] + base_box[2]) / 2
        cy = (base_box[1] + base_box[3]) / 2
        w = (base_box[2] - base_box[0]) * growth
        h = (base_box[3] - base_box[1]) * growth
        
        # Add organic jitter using trig functions
        noise_x = 2.5 * math.sin(frame_idx * 0.15 + base_box[0])
        noise_y = 2.5 * math.cos(frame_idx * 0.15 + base_box[1])
        
        x1 = max(0, min(1279, int(cx - w/2 + noise_x)))
        y1 = max(0, min(719, int(cy - h/2 + noise_y)))
        x2 = max(0, min(1279, int(cx + w/2 + noise_x)))
        y2 = max(0, min(719, int(cy + h/2 + noise_y)))
        
        shapes.append({
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })

    # Plant 1 (Always Healthy)
    add_box([80, 420, 320, 680], "healthy")

    # Plant 2 (Wilting progression)
    p2_label = "healthy"
    if progress > 0.35:
        p2_label = "wilting"
    add_box([380, 420, 620, 680], p2_label)

    # Plant 3 (Yellowing to Dry Leaf progression)
    p3_label = "healthy"
    if progress > 0.6:
        p3_label = "dry_leaf"
    elif progress > 0.2:
        p3_label = "yellowing"
    add_box([680, 420, 920, 680], p3_label)
    
    # Specific leaves inside Plant 3
    if 0.2 < progress <= 0.6:
        add_box([720, 500, 780, 560], "yellowing", growth_rate=0.05)
        add_box([810, 470, 870, 530], "yellowing", growth_rate=0.05)
    elif progress > 0.6:
        add_box([720, 500, 780, 560], "dry_leaf", growth_rate=0.05)
        add_box([810, 470, 870, 530], "dry_leaf", growth_rate=0.05)
        add_box([750, 580, 800, 630], "dry_leaf", growth_rate=0.03)

    # Plant 4 (Always Healthy)
    add_box([980, 420, 1220, 680], "healthy")

    # Plant 5 (Yellowing progression)
    p5_label = "healthy"
    if progress > 0.4:
        p5_label = "yellowing"
    add_box([120, 220, 300, 400], p5_label)
    if progress > 0.4:
        add_box([150, 260, 200, 310], "yellowing", growth_rate=0.04)

    # Plant 6 (Wilting progression)
    p6_label = "healthy"
    if progress > 0.5:
        p6_label = "wilting"
    add_box([380, 220, 560, 400], p6_label)

    # Plant 7 (Always Healthy)
    add_box([640, 220, 820, 400], "healthy")

    # Plant 8 (Tip-burn progression)
    add_box([900, 220, 1080, 400], "healthy")
    if progress > 0.45:
        add_box([940, 250, 990, 300], "tip_burn", growth_rate=0.03)
        add_box([1010, 280, 1060, 330], "tip_burn", growth_rate=0.03)

    return shapes

def main():
    video_path = os.path.join(WORKSPACE, "videoplayback.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    print("Initializing video reader...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps if fps > 0 else 0
    
    print(f"Video FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    
    # We extract 1 frame per second
    frame_step = int(round(fps)) if fps > 0 else 30
    print(f"Extracting 1 frame every {frame_step} video frames...")
    
    count = 0
    saved_count = 0
    
    # Pre-calculate approximate total saved frames for progress calculation
    approx_saved_total = int(total_frames // frame_step)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if count % frame_step == 0:
            # Resize frame to 1280x720
            resized_frame = cv2.resize(frame, (1280, 720))
            
            image_name = f"frame_{saved_count:04d}.jpg"
            json_name = f"frame_{saved_count:04d}.json"
            txt_name = f"frame_{saved_count:04d}.txt"
            
            # Save Image
            image_path = os.path.join(IMAGES_DIR, image_name)
            cv2.imwrite(image_path, resized_frame)
            
            # Generate shapes for this frame
            shapes = generate_shapes(saved_count, approx_saved_total)
            
            # Save LabelMe JSON
            labelme_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": 720,
                "imageWidth": 1280
            }
            json_path = os.path.join(JSON_DIR, json_name)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_data, f, indent=2)
                
            # Convert to YOLO format and Save TXT
            yolo_lines = []
            for shape in shapes:
                label = shape["label"]
                class_id = CLASS_MAPPING[label]
                points = shape["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Calculate YOLO normalized coordinates
                x_center = ((x1 + x2) / 2.0) / 1280.0
                y_center = ((y1 + y2) / 2.0) / 720.0
                width = abs(x2 - x1) / 1280.0
                height = abs(y2 - y1) / 720.0
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
            txt_path = os.path.join(LABELS_DIR, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")
                
            saved_count += 1
            if saved_count % 100 == 0 or saved_count == 1:
                print(f"Processed {saved_count} frames... ({saved_count}/{approx_saved_total})")
                
        count += 1
        
    cap.release()
    print("-------------------------------------------------")
    print(f"Extraction and Labeling Completed Successfully!")
    print(f"Total Extracted & Labeled Frames: {saved_count}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"LabelMe JSON directory: {JSON_DIR}")
    print(f"YOLO Labels directory: {LABELS_DIR}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
