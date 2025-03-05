import torch
from transformers import AutoImageProcessor, DetrForObjectDetection
from data_processing import get_hf_dataset, KITTIMOTS_CocoDetection
import evaluate
from tqdm import tqdm
import numpy as np

def collate_fn(batch):
    pixel_values = []
    for item in batch:
        img = item["pixel_values"]
        # Ensure the image is (C, H, W)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        pixel_values.append(img)
        
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    output = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }
    return output


# Set up the device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image processor and model, then move model to the device
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.to(device)
model.eval()  # set model to evaluation mode

# Define dataset path and load your preprocessed Hugging Face dataset
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
train_instances_ids = [19, 20, 9, 7, 1, 8, 15, 11, 13, 18, 4, 5]
test_instances_ids = [10]  #[10, 6, 2, 16, 0, 17, 3, 14, 12]


hf_dataset = KITTIMOTS_CocoDetection(dataset_path, instances_ids=test_instances_ids)
print(len(hf_dataset))
print("HF Dataset loaded in evaluation")

# Load the COCO evaluation module (ensuring predictions/references are in expected format)
coco_eval = evaluate.load("ybelkada/cocoevaluate", coco=hf_dataset.coco)
print("Evaluator loaded")

val_dataloader = torch.utils.data.DataLoader(
    hf_dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn
)


with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        labels = [
            {k: v for k, v in t[0].items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
      
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = image_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if label.item() not in [1, 3]:
                    #continue
                    pass

                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} ({label.item()}) with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                
        coco_eval.add(prediction=results, reference=labels)
        del batch

results = coco_eval.compute()
print(results)