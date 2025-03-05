import torch
import os, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dataset import build_kitti_mots_dicts


dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
output_path = "/ghome/c5mcv07/C5_G7_MCV/Task1/faster-rcnn/output/pre_trained_inference"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_dicts = build_kitti_mots_dicts(dataset_path)

DatasetCatalog.register("kitti_mots", lambda: build_kitti_mots_dicts("/home/mcv/datasets/C5/KITTI-MOTS", instances_ids=[1, 2, 3]))
MetadataCatalog.get("kitti_mots").set(thing_classes=["pedestrian", "car"])

kitti_mots_metadata = MetadataCatalog.get("kitti_mots")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
# Output folder
cfg.OUTPUT_DIR = output_path
cfg.MODEL.DEVICE = device

# Run inference on all images
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("kitti_mots", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "kitti_mots")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
