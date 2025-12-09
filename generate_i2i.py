import os
import os.path as osp

import torch
import gc
from transformers import Mistral3ForConditionalGeneration

from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
from diffusers.utils import load_image
from peft import cast_mixed_precision_params, PeftModel
from transformers import BitsAndBytesConfig

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

is_lora_test = True
skip_text_encoder = True

# --- Prepare Prompt Embeddings ---

if skip_text_encoder:

  print("Using cached prompt embeddings.")

  # Load pre-computed text embeddings
  prompt_embeds = torch.load("cache/prompt_embeds_000001.pt").unsqueeze(0)

else:

  text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, subfolder="text_encoder", dtype=torch_dtype, device_map="cpu"
  )
  pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=None, vae=None, text_encoder=text_encoder, torch_dtype=torch_dtype
  )
  pipe.enable_model_cpu_offload()
  print("Text Encoder loaded.")
  print_gpu_memory_usage()

  prompt = "Create a studio fashion photograph of the subject from Image 1 with their exact face, hair, and body shape. The subject should be wearing the clothing from Image 2, which should fit perfectly and match the style and color. The pose should be exactly as shown in Image 3. The background should be a plain white or very light grey, typical of a professional studio setting. Use high-quality studio lighting to ensure the subject and clothing are well-lit and the details are sharp."
  prompt = """Minimalist and clean contemporary fashion photography, often used for e-commerce lookbooks. The focus is entirely on the model and the apparel.
  The subject is wearing garment from the clothing reference image.
  The subject is posing exactly as shown in the reference pose image.
  Strictly preserve original facial identity, hairstyle, and body physique."""

  with torch.no_grad():
    prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=pipe._execution_device,
        max_sequence_length=128 if is_lora_test else 512
    )[0].cpu()

  print("Prompt generated. Text Encoder run is complete.")
  del text_encoder
  del pipe
  gc.collect()
  torch.cuda.empty_cache()
  print_gpu_memory_usage()

# --- Generate image ---

transformer = Flux2Transformer2DModel.from_pretrained(
  repo_id, subfolder="transformer", torch_dtype=torch_dtype,
)
vae = AutoencoderKLFlux2.from_pretrained(
  repo_id, subfolder="vae", torch_dtype=torch_dtype,
  device_map=device
)

pipe = Flux2Pipeline.from_pretrained(
  repo_id,
  transformer=transformer,
  vae=vae,
  text_encoder=None,
  torch_dtype=torch_dtype
).to(device)
transformer.set_attention_backend("flash_hub")

if is_lora_test:
  print("Loading LoRA")

  transformer.load_lora_adapter(
    "output/lora",
    weight_name="pytorch_lora_weights.safetensors",
    autocast_adapter_dtype=False
  )

  transformer.fuse_lora()
  transformer.unload_lora()

  print("LoRA fused.")

print("DiT loaded.")
print_gpu_memory_usage()

INPUT_DIR = "data/image"
OUTPUT_DIR = "output/"

input_images = [x for x in os.listdir(INPUT_DIR) if osp.splitext(x)[1] in (".png", ".jpg", ".webp")]
if not osp.isdir(OUTPUT_DIR):
   os.makedirs(OUTPUT_DIR)

for imgname in input_images:
  cond_images = [
    load_image("data/input/028.jpg"),
    load_image("data/cloth/014.jpg"),
    load_image(osp.join(INPUT_DIR, imgname)).resize((320, 400)),
  ]

  image = pipe(
    image=cond_images,
    prompt_embeds=prompt_embeds.to(device),
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=28, # 50 default, 28 is a good trade-off
    guidance_scale=4,
    height=1024,
    width=800,
  ).images[0]

  image.save(osp.join(OUTPUT_DIR, imgname))

print("Images generated. Flux2 run is complete.")
del transformer
del vae
del pipe
gc.collect()
torch.cuda.empty_cache()
print_gpu_memory_usage()

