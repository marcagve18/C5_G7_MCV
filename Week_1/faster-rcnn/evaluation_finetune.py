import torch
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dataset import build_kitti_mots_dicts
from predictor import CustomPredictor


dataset_path = Path("/home/mcv/datasets/C5/KITTI-MOTS")
output_path = Path("/ghome/c5mcv07/C5_G7_MCV/Week_1/faster-rcnn/output/pre_trained_inference")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

### FINETUNE

train_instances_ids = [19, 20, 9, 7, 1, 8, 15, 11, 13, 18, 4, 5]
test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]

for split in ["train", "test"]:
    instances_ids = train_instances_ids if split == "train" else test_instances_ids
    DatasetCatalog.register(f"kitti_mots_{split}", lambda: build_kitti_mots_dicts("/home/mcv/datasets/C5/KITTI-MOTS", instances_ids=instances_ids))
    MetadataCatalog.get(f"kitti_mots_{split}").set(thing_classes=["person", "bicycle", "car"])
kitti_mots_metadata = MetadataCatalog.get("kitti_mots_train")

cfg = get_cfg()
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = str(output_path / "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# Output folder
cfg.OUTPUT_DIR = str(output_path)
cfg.MODEL.DEVICE = device

# Run inference on all images
output_path = Path(cfg.OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("kitti_mots", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "kitti_mots")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
