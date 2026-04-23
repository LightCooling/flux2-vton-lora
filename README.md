# FLUX.2 Virtual Try-On LoRA

LoRA fine-tuning experiments for shifting `diffusers/FLUX.2-dev-bnb-4bit` toward a virtual try-on (VTON) domain under consumer GPU limits.

The repository keeps the training path intentionally small: text encoder and VAE outputs are precomputed into a dataset cache, then `train.py` trains only the FLUX.2 transformer LoRA weights from those tensors.

## Design Notes

- Training was designed around an RTX 4090 class GPU.
- Encoders are intentionally removed from the training loop and replaced by cached prompt embeddings and VAE latents.
- Validation generation is intentionally not part of `train.py`; it would require loading extra pipeline pieces and increases VRAM pressure.
- Prompt embeddings default to `128` tokens to reduce memory use. Keep prompts short and specific.
- `generate_i2i.py` represents the target VTON inference workflow and may use a richer condition stack than the compact training cache.

## Repository Layout

- `prepare_cache.py` - builds cached prompt embeddings, text ids, target latents, and two condition latent tensors.
- `train.py` - trains FLUX.2 LoRA weights from the precomputed cache.
- `generate_i2i.py` - target image-to-image VTON inference script.
- `generate_t2i.py` - small text-to-image smoke script.
- `AGENTS.md` - project audit and maintenance notes.

Large local artifacts are ignored by git: `data/`, `cache/`, `output/`, `.venv/`, notebooks, and ad-hoc experiment scripts.

## Requirements

- Python `>=3.11`
- CUDA-capable PyTorch environment for practical training/inference
- `uv` for environment management
- Access to `diffusers/FLUX.2-dev-bnb-4bit`

The `diffusers` source is pinned in `pyproject.toml` to the commit used by this project because FLUX.2 helper methods used here are private APIs.

## Setup

```bash
uv sync
```

If you use a manually managed environment, install the dependencies from `pyproject.toml` and make sure the pinned `diffusers` revision is used.

## Prepare Cache

```bash
uv run python prepare_cache.py \
  --pretrained_model_name_or_path diffusers/FLUX.2-dev-bnb-4bit \
  --dataset_name data/dataset \
  --image_column image \
  --cond1_image_column cloth \
  --cond2_image_column densepose \
  --caption_column caption \
  --output_dir cache \
  --resolution 512 \
  --max_sequence_length 128
```

`prepare_cache.py` refuses to write into a non-empty cache directory. Use `--overwrite_cache` only when you intentionally want to delete and rebuild the cache.

Expected cached tensor shapes for the current 384x512-style setup are:

- `prompt_embeds`: `(128, 15360)`
- `text_ids`: `(128, 4)`
- `latents`, `cond1_latents`, `cond2_latents`: `(128, 32, 24)`

## Train

```bash
uv run python train.py \
  --pretrained_model_name_or_path diffusers/FLUX.2-dev-bnb-4bit \
  --cache_dir cache \
  --output_dir output/lora \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --mixed_precision bf16
```

`train.py` validates that cache indices are contiguous and that the first cached tensors have the expected rank, dtype family, and matching shapes before loading the model.

## Inference

```bash
uv run python generate_i2i.py
```

The inference script currently uses local demo paths under `data/` and writes images to `output/`. Update those paths in the script for your local VTON run.

## Maintenance

Before publishing changes:

```bash
uv run python -m py_compile prepare_cache.py train.py generate_i2i.py generate_t2i.py
git status --short
```

Keep generated caches, LoRA outputs, downloaded datasets, notebooks, and one-off experiments out of git.
