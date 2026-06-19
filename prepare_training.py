import os
import shutil

WORKSPACE = r"c:\Users\SARANKUMAR\Downloads\AgriVision_AI"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
IMAGES_SRC = os.path.join(DATASET_DIR, "images")
LABELS_SRC = os.path.join(DATASET_DIR, "labels")

# Target folders
TRAIN_IMG = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL = os.path.join(DATASET_DIR, "val", "labels")

# Create folders
os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# List all extracted frames
all_images = sorted([f for f in os.listdir(IMAGES_SRC) if f.endswith('.jpg')])

print(f"Total images found: {len(all_images)}")
print("Performing interleaved dataset split (every 5th frame to validation)...")

for img_file in all_images:
    base_name = os.path.splitext(img_file)[0]
    txt_file = base_name + ".txt"
    
    # Paths
    img_path = os.path.join(IMAGES_SRC, img_file)
    txt_path = os.path.join(LABELS_SRC, txt_file)
    
    # Interleaved split: every 5th frame goes to validation
    frame_idx = int(base_name.split('_')[1])
    
    if frame_idx % 5 == 0:
        # Move to validation
        shutil.move(img_path, os.path.join(VAL_IMG, img_file))
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(VAL_LBL, txt_file))
    else:
        # Move to training
        shutil.move(img_path, os.path.join(TRAIN_IMG, img_file))
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(TRAIN_LBL, txt_file))

print("Dataset successfully split into train and val folders!")

# Generate data.yaml with forward slashes (required for YOLO cross-platform path handling)
yaml_content = f"""path: {DATASET_DIR.replace(chr(92), '/')}
train: train/images
val: val/images

nc: 5
names:
  0: healthy
  1: wilting
  2: yellowing
  3: tip_burn
  4: dry_leaf
"""

yaml_path = os.path.join(WORKSPACE, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Generated data.yaml at {yaml_path}")
