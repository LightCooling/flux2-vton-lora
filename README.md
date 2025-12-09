# FLUX.2 Virtual Try-On LoRA

This project implements LoRA fine-tuning for FLUX.2 text-to-image/image-to-image model to improve virtual try-on (VTON) capabilities using image-to-image generation. The training implementation builds upon the diffusers library's Flux.2 DreamBooth+LoRA training example.

## Motivation

Due to the large size of FLUX.2 model that exceed consumer GPU VRAM capacities, pre-computed dataset caches were employed to optimize resource usage. While CPU offloading is possible, it inefficiently utilizes excessive RAM with minimal performance advantages. Training was conducted on an RTX 4090 GPU, constraining resolution to 384x512 pixels for training and limiting inference to 768x1024 pixels. To further accommodate model constraints, prompt embeddings were reduced from the default 512 tokens to 128 tokens, necessitating careful prompt tuning to maintain effectiveness within this shortened context window.

## Features

- Text-to-image generation (`generate_t2i.py`)
- Image-to-image virtual try-on with LoRA (`generate_i2i.py`)
- Cache preparation for efficient training (`prepare_cache.py`)
- LoRA training script for FLUX.2 (`train.py`)

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- PEFT
- Transformers
- PIL
- Torchvision

## Usage

1. Prepare caches: `python prepare_cache.py --pretrained_model_name_or_path diffusers/FLUX.2-dev-bnb-4bit --dataset_name your_dataset`
2. Train LoRA: `python train.py --pretrained_model_name_or_path diffusers/FLUX.2-dev-bnb-4bit --cache_dir cache`
3. Generate images: `python generate_t2i.py` or `python generate_i2i.py`

## Current Status

WIP - Work in progress
