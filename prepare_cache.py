#!/usr/bin/env python
# coding=utf-8
# Training script for Flux.2 I2I caching

import argparse
import os
import shutil

import torch
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset

from diffusers import AutoencoderKLFlux2, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor


class SimpleFluxDataset(Dataset):
    def __init__(self, dataset_name, image_column, cond1_column, cond2_column, caption_column, resolution):
        self.dataset = load_dataset(dataset_name)["train"]
        self.length = len(self.dataset)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((resolution, resolution // 4 * 3)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_column = image_column
        self.cond1_column = cond1_column
        self.cond2_column = cond2_column
        self.caption_column = caption_column

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process target image
        image = item[self.image_column]
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = self.transform(image)

        # Process condition1 image
        cond1 = item[self.cond1_column]
        cond1 = exif_transpose(cond1)
        if cond1.mode != "RGB":
            cond1 = cond1.convert("RGB")
        cond1_values = self.transform(cond1)

        # Process condition2 image
        cond2 = item[self.cond2_column]
        cond2 = exif_transpose(cond2)
        if cond2.mode != "RGB":
            cond2 = cond2.convert("RGB")
        cond2_values = self.transform(cond2)

        caption = item[self.caption_column]

        return {
            "pixel_values": pixel_values,
            "cond1_values": cond1_values,
            "cond2_values": cond2_values,
            "prompts": caption,
        }


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Cache preparation script for Flux.2 I2I training.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained Flux.2 model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Target image column",
    )
    parser.add_argument(
        "--cond1_image_column",
        type=str,
        default="cloth",
        help="Condition1 image column",
    )
    parser.add_argument(
        "--cond2_image_column",
        type=str,
        default="densepose",
        help="Condition2 image column",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="Caption column",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache",
        help="Output cache directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Delete and recreate output_dir if it already contains files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=128,
        help="Maximum sequence length to use with the text encoder",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def free_memory():
    # Clear GPU memory
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main(args):
    print("Loading dataset...")
    train_dataset = SimpleFluxDataset(
        args.dataset_name,
        args.image_column,
        args.cond1_image_column,
        args.cond2_image_column,
        args.caption_column,
        args.resolution,
    )

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        if not args.overwrite_cache:
            raise ValueError(
                f"Output cache directory is not empty: {args.output_dir}. "
                "Use --overwrite_cache to delete it before writing a new cache."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Text embeddings
    print("Loading text encoder...")
    weight_dtype = torch.bfloat16

    tokenizer = PixtralProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype
    )
    text_encoder.requires_grad_(False)

    text_encoding_pipeline = Flux2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoding_pipeline.to(device)

    print("Processing text embeddings...")

    # Prompts encode
    for i in tqdm(range(len(train_dataset))):
        item = train_dataset[i]
        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            item["prompts"],
            max_sequence_length=args.max_sequence_length
        )
        torch.save(prompt_embeds.squeeze(0), os.path.join(args.output_dir, f"prompt_embeds_{i:06d}.pt"))
        torch.save(text_ids.squeeze(0), os.path.join(args.output_dir, f"text_ids_{i:06d}.pt"))

    # Free memory
    del text_encoder, tokenizer, text_encoding_pipeline
    free_memory()

    # Phase 2: VAE latents
    print("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    vae.to(device)

    print("Processing image latents...")

    # VAE encode
    for i in tqdm(range(len(train_dataset))):
        item = train_dataset[i]
        for column_name, file_prefix in [
            ("pixel_values", "latents"),
            ("cond1_values", "cond1_latents"),
            ("cond2_values", "cond2_latents")
        ]:
            image = item[column_name].unsqueeze(0).to(device, dtype=vae.dtype)
            latents = vae.encode(image).latent_dist.mode()
            latents = Flux2Pipeline._patchify_latents(latents)
            latents = (latents - latents_bn_mean.to(device)) / latents_bn_std.to(device)
            torch.save(latents.squeeze(0).cpu(), os.path.join(args.output_dir, f"{file_prefix}_{i:06d}.pt"))

    print(f"Caches saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
