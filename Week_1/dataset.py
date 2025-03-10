import os
import shutil
from pycocotools import mask as maskUtils

# Paths
KITTI_MOTS_PATH = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS"
IMAGES_PATH = os.path.join(KITTI_MOTS_PATH, "training/image_02")  # Image sequences
INSTANCES_TXT_PATH = os.path.join(KITTI_MOTS_PATH, "instances_txt")  # Correct path for annotations
OUTPUT_PATH = "/home/c5mcv07/C5_G7_MCV/Week_1/yolov11/src"  # Output inside YOLO folder

# Define train and validation sequences manually
TRAIN_SEQUENCES = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]  # Modify these as needed
VAL_SEQUENCES = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]  # Modify these as needed

# Create dataset structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_PATH, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, f"labels/{split}"), exist_ok=True)

# KITTI to COCO mapping (updated for 3 classes)
KITTI_TO_COCO = {1: 2, 2: 0}

# Log missing images
missing_images_log = os.path.join(OUTPUT_PATH, "missing_images.log")

# Function to process a sequence
def process_sequence(seq_id, split):
    seq_image_dir = os.path.join(IMAGES_PATH, seq_id)
    seq_annot_file = os.path.join(INSTANCES_TXT_PATH, f"{seq_id}.txt")  # Corrected path

    if not os.path.exists(seq_annot_file):
        print(f"❌ Annotation file missing for sequence {seq_id}. Skipping.")
        return

    # Create sequence folder for images and labels
    seq_image_output_dir = os.path.join(OUTPUT_PATH, f"images/{split}/{seq_id}")
    seq_label_output_dir = os.path.join(OUTPUT_PATH, f"labels/{split}/{seq_id}")
    
    os.makedirs(seq_image_output_dir, exist_ok=True)
    os.makedirs(seq_label_output_dir, exist_ok=True)

    with open(seq_annot_file, "r") as f:
        for line in f:
            try:
                frame, obj_id, class_id, height, width, rle = line.strip().split(" ", 5)
                frame_id, obj_id, class_id = int(frame), int(obj_id), int(class_id)

                if class_id == 10:  # Ignore unlabeled
                    continue

                if class_id in KITTI_TO_COCO:
                    class_id = KITTI_TO_COCO[class_id]
                else:
                    continue  # Skip unknown classes

                # Convert RLE to bounding box
                height, width = int(height), int(width)
                rle_obj = {'counts': rle, 'size': [height, width]}
                bbox = [int(coord) for coord in maskUtils.toBbox(rle_obj)]
                x, y, w, h = bbox

                # Normalize for YOLO format
                x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
                norm_w, norm_h = w / width, h / height
                annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

                # Image file
                image_file = f"{frame_id:06d}.png"
                src_image_path = os.path.join(seq_image_dir, image_file)
                dest_image_path = os.path.join(seq_image_output_dir, image_file)

                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dest_image_path)

                    # Write annotation
                    annotation_file = os.path.join(seq_label_output_dir, f"{frame_id:06d}.txt")
                    with open(annotation_file, "a") as ann_f:
                        ann_f.write(annotation + "\n")

                else:
                    print(f"⚠️ Image file {image_file} missing in {seq_id}. Logging.")
                    with open(missing_images_log, "a") as log_f:
                        log_f.write(f"{seq_id}/{image_file}\n")

            except Exception as e:
                print(f"❌ Error processing {seq_id}, line: {line.strip()} - {e}")

# Process sequences manually
for seq in TRAIN_SEQUENCES:
    process_sequence(seq, "train")

for seq in VAL_SEQUENCES:
    process_sequence(seq, "val")

# Create YOLO data.yaml
data_yaml = f"""train: {OUTPUT_PATH}/images/train
val: {OUTPUT_PATH}/images/val
nc: 2
names: ['person', 'car']
"""

with open(os.path.join(OUTPUT_PATH, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("✅ YOLO dataset created inside:", OUTPUT_PATH)
