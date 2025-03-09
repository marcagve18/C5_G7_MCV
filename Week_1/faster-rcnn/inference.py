import torch
from pathlib import Path
import cv2
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from dataset import build_kitti_mots_dicts
from predictor import CustomPredictor


dataset_path = Path("/home/mcv/datasets/C5/KITTI-MOTS")
output_path = Path("/ghome/c5mcv07/C5_G7_MCV/Week_1/faster-rcnn/output/pre_trained_inference")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

DatasetCatalog.register("kitti_mots", lambda: build_kitti_mots_dicts("/home/mcv/datasets/C5/KITTI-MOTS", instances_ids=[1, 2, 3]))

dataset_dicts = build_kitti_mots_dicts(str(dataset_path))
kitti_mots_metadata = MetadataCatalog.get("kitti_mots")
kitti_mots_metadata.set(thing_classes=["person", "bicycle", "car"])

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
# Output folder
cfg.OUTPUT_DIR = str(output_path)
cfg.MODEL.DEVICE = device

# Run inference on all images
output_path = Path(cfg.OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
predictor = CustomPredictor(cfg)

for d in tqdm(dataset_dicts):
    file_name = Path(d["file_name"])
    image_id = d["image_id"]

    output_file = output_path / file_name.parent.name / file_name.name
    output_file.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(file_name))[:, :, ::-1]
    outputs = predictor(img)  # Run inference
    instances = outputs["instances"]

    # Use the filtered instances for visualization
    v = Visualizer(img, kitti_mots_metadata, scale=0.5)
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2.imwrite(str(output_file), out.get_image())
