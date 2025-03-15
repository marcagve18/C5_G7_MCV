from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from huggingface_hub import login
from dotenv import load_dotenv
import os 

load_dotenv()
HF_TOKEN = os.getenv('HF_MARC')
login(HF_TOKEN)


checkpoint = "/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000"
# Load your finetuned model and its processor (adjust paths as needed)
model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
processor = AutoImageProcessor.from_pretrained(checkpoint)

# Optionally, set your model card description
model.model_card_data = {
    "language": "en",
    "license": "apache-2.0",
    "tags": ["instance-segmentation", "mask2former", "finetuned"],
    "description": "Finetuned Mask2Former model focusing on improving predictions for cars and persons."
}

# Push the model to the Hub
model.push_to_hub("marcagve18/mask2former-kittimots-instance-seg")
processor.push_to_hub("marcagve18/mask2former-kittimots-instance-seg")

print("Model and processor successfully pushed to the Hugging Face Hub!")