import torch
from tqdm import tqdm
import cv2
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode, Visualizer

from dataset import build_balloon_dicts


dataset_path = Path("/ghome/c5mcv07/C5_G7_MCV/Week_2/balloon")
output_path = Path("/ghome/c5mcv07/C5_G7_MCV/Week_2/faster-rcnn/output/train_domain_shift")

output_inference_path = output_path / "inference"
output_inference_path.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

### FINETUNE

for split in ["train", "val"]:
    DatasetCatalog.register(f"balloon_{split}", lambda: build_balloon_dicts(str(dataset_path / split)))
    MetadataCatalog.get(f"balloon_{split}").set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# Update datasets
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ("balloon_val",)
cfg.DATALOADER.NUM_WORKERS = 2
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = ""  # TRY TRAINING THE WHOLE MODEL
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 10.0  # You can experiment with the clipping value
cfg.SOLVER.WEIGHT_DECAY = 0.0001
# Output folder
cfg.OUTPUT_DIR = str(output_path)
cfg.MODEL.DEVICE = device

# Run inference on all images
output_path = Path(cfg.OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = str(output_path / "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# Output folder
cfg.OUTPUT_DIR = str(output_path)
cfg.MODEL.DEVICE = device
predictor = DefaultPredictor(cfg)

# Save inference

dataset_dicts = build_balloon_dicts(str(dataset_path / "val"))
for d in tqdm(dataset_dicts):
    file_name = Path(d["file_name"])
    im = cv2.imread(d["file_name"])[:, :, ::-1]
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im,
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_file = output_inference_path / file_name.name
    cv2.imwrite(str(output_file), out.get_image()[:, :, ::-1])

# Evaluate finetune

evaluator = COCOEvaluator("balloon_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "balloon_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
