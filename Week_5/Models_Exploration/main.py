import PIL.Image
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, AutoPipelineForText2Image, DiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
import PIL
from typing import List, Optional, Dict, Any
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
import time 

load_dotenv()
HF_TOKEN = os.getenv('HF_MARC') # Use a generic name or your specific one
if HF_TOKEN:
    login(HF_TOKEN)
else:
    print("Warning: Hugging Face token not found. Ensure you are logged in if required for private models.")

# Consider using a dictionary for easier management and adding metadata like default resolution
MODEL_INFO = {
    "SD15": {
        "hf_name": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "pipeline_class": StableDiffusionPipeline,
    },
    "SD21": {
        "hf_name": "stabilityai/stable-diffusion-2-1",
        "pipeline_class": StableDiffusionPipeline,
    },
    "SD21Turbo": {
        "hf_name": "stabilityai/sd-turbo",
        "pipeline_class": StableDiffusionPipeline,
        "inference_params": {
            "guidance_scale": 0,
            "num_inference_steps": 1,
        } 
    },
    "SDXL": {
        "hf_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": DiffusionPipeline, 
    },
    "SDXLTurbo": {
        "hf_name": "stabilityai/sdxl-turbo",
        "pipeline_class": AutoPipelineForText2Image,
        "inference_params": {
            "guidance_scale": 0,
            "num_inference_steps": 1,
        }
    },
    "SD3Medium": { # Renamed for clarity, matching HF name better
        "hf_name": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_class": StableDiffusion3Pipeline,
    },
}

ALLOWED_MODELS = list(MODEL_INFO.keys())

class DiffusionModel:
    """
    A simple wrapper for various Stable Diffusion models from Hugging Face diffusers.
    """
    def __init__(self, model_name: str, device: str = 'cuda'):
        if model_name not in ALLOWED_MODELS:
            allowed_str = ", ".join(ALLOWED_MODELS)
            raise ValueError(f"Model `{model_name}` is invalid. Allowed models are: {allowed_str}")

        self.model_name = model_name
        self.device = device
        self.info = MODEL_INFO[model_name]
        self.pipeline = None

        print(f"Loading {self.model_name} ({self.info['hf_name']})")

        pipeline_options: Dict[str, Any] = {}

        # Use float16 for GPU acceleration and memory saving
        if self.device == 'cuda':
            pipeline_options['torch_dtype'] = torch.float16

        PipelineClass = self.info['pipeline_class']
        self.pipeline = PipelineClass.from_pretrained(
            self.info['hf_name'],
            safety_checker=None,
            **pipeline_options
        )

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        if model_name == "SD21":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)


        self.pipeline.to(self.device)
        print(self.pipeline)
        print(f"{self.model_name} loaded successfully on {self.device}.")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = 42,
    ) -> List[PIL.Image.Image]:
        """
        Generates images based on the prompt using the loaded model.

        Args:
            prompt (str): The text prompt to guide image generation.
            negative_prompt (Optional[str]): Text prompt to guide *away* from.
            num_inference_steps (Optional[int]): Number of denoising steps. Defaults to model-specifics.
            guidance_scale (Optional[float]): Classifier-Free Guidance scale. Defaults to model-specifics.
            height (Optional[int]): Height of the generated image. Defaults to model's preferred resolution.
            width (Optional[int]): Width of the generated image. Defaults to model's preferred resolution.
            num_images_per_prompt (int): Number of images to generate for the prompt. Defaults to 1.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            List[PIL.Image.Image]: A list of generated PIL Images.
        """
        if self.pipeline is None:
             raise RuntimeError("Pipeline is not loaded. Initialize the class first.")

        # --- Prepare Generator for Seed ---
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- Log Generation Parameters ---
        print(f"Generating with {self.model_name}:")
        print(f"  Prompt: '{prompt}'")
        if negative_prompt: print(f"  Negative Prompt: '{negative_prompt}'")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Seed: {seed if seed is not None else 'Random'}")
        print(f"  Num Images: {num_images_per_prompt}")

        if 'inference_params' in self.info:
            num_inference_steps = self.info['inference_params']['num_inference_steps']
            guidance_scale = self.info['inference_params']['guidance_scale']

        # --- Run Inference ---
        start_time = time.time()
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        ) 

        images: List[PIL.Image.Image] = output.images

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        return images

    def unload(self):
        if self.pipeline is not None:
            print(f"Unloading {self.model_name} from {self.device}...")
            try:
                 self.pipeline.to('cpu')
            except Exception as e:
                 print(f"Could not move pipeline to CPU: {e}")
            del self.pipeline
            self.pipeline = None
            if self.device == 'cuda':
                torch.cuda.empty_cache() 
            print(f"{self.model_name} unloaded.")
        else:
            print(f"{self.model_name} is not currently loaded.")


def check_models(test_device='cpu'):
    """
    Tries to load each allowed model to check availability and basic loading.
    """
    print(f"\n--- Checking Model Availability (using device: {test_device}) ---")
    successful_models = []
    failed_models = []

    for model_key in ALLOWED_MODELS:
        m = None
        try:
            print("-" * 20)
            m = DiffusionModel(model_key, device=test_device)
            successful_models.append(model_key)
            m.unload()
            del m 
            if test_device == 'cuda':
                torch.cuda.empty_cache() 

        except Exception as e:
            print(f"\n!!! Failed to load model '{model_key}' !!!")
            print(f"Error: {e}\n")
            failed_models.append(model_key)
            if m is not None:
                m.unload()
                del m
            if test_device == 'cuda':
                 torch.cuda.empty_cache()

    print("-" * 20)
    print("\n--- Model Check Summary ---")
    if successful_models:
        print(f"Successfully loaded: {', '.join(successful_models)}")
    if failed_models:
        print(f"Failed to load: {', '.join(failed_models)}")
    else:
        print("All allowed models were loaded successfully!")
    print("-" * 20)

'''
TESTS TO VERIFY THAT THE MODELS LOAD CORRECTLY. IF YOU ADD A NEW ONE,
UNCOMMENT THE FOLLOWING LINES TO ENSURE PROPER BEHAVIOUR.
Set test_device='cuda' if you want to check GPU loading specifically (needs enough VRAM!)
'''
#check_models(test_device='cpu') # Safer check on CPU first

# Determine device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available(): 
     device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

TEST_PROMPTS = [
    "plate of spaghetti, with tomato sauce, high-quality, HD, 4K, realistic, high-res",
    "plate of steak with french fries, high-quality, HD, 4K, realistic, high-res",
    "plate with nachos and guacamole, high resolution, HD, realistic",
    "plate of soup with pasta and beef, high-quality, HD, 4K, realistic, high-res",
]

# --- Actual Generation Example ---
if __name__ == "__main__":
    NUM_INF_STEPS = [5, 15, 25, 35, 45, 50]
    CFGS = [3, 5, 7.5, 10, 12, 15]
    params = [""]

    for model_to_test in ["SD15", "SDXLTurbo"]:
        diff_model = DiffusionModel(model_name=model_to_test, device=device)
        for param in params:
                if model_to_test not in ALLOWED_MODELS:
                    print(f"Model {model_to_test} not in ALLOWED_MODELS. Please choose a valid model.")
                else:
                    print(f"\n--- Running generation test with {model_to_test} ---")

                    for prompt in TEST_PROMPTS:
                        neg_prompt = param
                        # Use model's default resolution by not specifying height/width
                        # Or override: height=512, width=768,
                        images = diff_model.generate(
                            prompt=prompt,
                            negative_prompt="low quality, blurry, cartoon, drawing, illustration, sketch",
                            seed=42, # For reproducibility
                            num_images_per_prompt=1,
                            num_inference_steps=25,
                            guidance_scale=7.5
                        )

                        if images:
                            # Save or display the image
                            for idx, image in enumerate(images):
                                save_path = f"generated_imgs/{model_to_test}_{prompt}_{idx}_scheduler_{diff_model.pipeline.scheduler}.png"
                                image.save(save_path)
                                print(f"Image saved to {save_path}")
                            # images[0].show() # Uncomment to display directly if environment supports it
                        else:
                                print("Image generation failed.")

        # Unload the model to free VRAM
        diff_model.unload()
