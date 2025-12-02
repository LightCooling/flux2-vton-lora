import torch
import gc
from transformers import Mistral3ForConditionalGeneration

from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKLFlux2


def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

# --- Prepare Prompt Embeddings ---

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
  repo_id, subfolder="text_encoder", dtype=torch_dtype, device_map="cpu"
)
pipe = Flux2Pipeline.from_pretrained(
  repo_id, transformer=None, vae=None, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()
print("Text Encoder loaded.")
print_gpu_memory_usage()

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

with torch.no_grad():
  prompt_embeds = pipe.encode_prompt(
      prompt=prompt,
      device=pipe._execution_device,
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
  repo_id, transformer=transformer, vae=vae, text_encoder=None, torch_dtype=torch_dtype
)
pipe.transformer.set_attention_backend("flash_hub")
print("DiT loaded.")
print_gpu_memory_usage()

image = pipe(
  prompt_embeds=prompt_embeds.to(device),
  generator=torch.Generator(device=device).manual_seed(42),
  num_inference_steps=28, # 50 default, 28 is a good trade-off
  guidance_scale=4,
  # height=512,
  # width=512,
).images[0]

print("Image generated. Flux2 run is complete.")
del transformer
del vae
del pipe
gc.collect()
torch.cuda.empty_cache()
print_gpu_memory_usage()

image.save("flux2_t2i_nf4.png")