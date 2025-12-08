#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Simplified Flux.2 LoRA training script using pre-computed caches

import argparse
import os
import random
import shutil

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from diffusers import (
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
from diffusers.utils import check_min_version
check_min_version("0.36.0.dev0")


class SimpleFluxDataset(Dataset):
    """Simple dataset that loads pre-computed caches"""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        # Determine length by counting prompt_embeds files
        prompt_embeds_files = [f for f in os.listdir(cache_dir) if f.startswith("prompt_embeds_") and f.endswith(".pt")]
        self.length = len(prompt_embeds_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "prompt_embeds": torch.load(os.path.join(self.cache_dir, f"prompt_embeds_{idx:06d}.pt")),
            "text_ids": torch.load(os.path.join(self.cache_dir, f"text_ids_{idx:06d}.pt")),
            "latents": torch.load(os.path.join(self.cache_dir, f"latents_{idx:06d}.pt")),
            "cond1_latents": torch.load(os.path.join(self.cache_dir, f"cond1_latents_{idx:06d}.pt")),
            "cond2_latents": torch.load(os.path.join(self.cache_dir, f"cond2_latents_{idx:06d}.pt")),
        }


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple Flux.2 LoRA training script.")

    # --- Base args ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory containing pre-computed caches",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # --- Training args ---
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file for bitsandbytes quantization.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )

    # --- LR args ---
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for Flux")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor for polynomial scheduler")

    # --- Optimizer args ---
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay for AdamW")
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for Adam.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for Adam.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # --- LoRA args ---
    parser.add_argument("--rank", type=int, default=4, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="The transformer modules to apply LoRA training on.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )

    # Load or create quantization config
    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        import json
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    # Load transformer
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )

    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    # Freeze base models
    transformer.requires_grad_(False)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Setup LoRA
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Move models to device
    transformer.to(device)

    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    # Setup optimizer
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes: pip install bitsandbytes")

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Cast to fp32 if needed
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Load dataset
    train_dataset = SimpleFluxDataset(args.cache_dir)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Calculate steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Setup LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps")

    for epoch in range(args.num_train_epochs):
        transformer.train()

        # torch.cuda.memory._record_memory_history(max_entries=100_000)

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype)
            text_ids = batch["text_ids"].to(device, dtype=torch.long)
            latents = batch["latents"].to(device, dtype=weight_dtype)
            cond1_latents = batch["cond1_latents"].to(device, dtype=weight_dtype)
            cond2_latents = batch["cond2_latents"].to(device, dtype=weight_dtype)

            if global_step >= args.max_train_steps:
                break

            with torch.enable_grad():
                bsz = latents.shape[0]

                # Sample noise and timesteps
                noise = torch.randn_like(latents)

                # Uniform timestep sampling
                u = torch.rand(bsz, device='cpu')
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device)

                # Add noise according to flow matching
                all_sigmas = torch.tensor(noise_scheduler.sigmas, device=device)
                sigmas = all_sigmas[indices].flatten()
                while len(sigmas.shape) < latents.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Prepare latent IDs
                cond_ids = Flux2Pipeline._prepare_image_ids([cond1_latents[0], cond2_latents[0]]).to(device)
                cond_ids = cond_ids.expand(bsz, -1, -1)
                latent_ids = Flux2Pipeline._prepare_latent_ids(latents).to(device)
                latent_ids = torch.cat([latent_ids, cond_ids], dim=1)

                # Pack conditions
                packed_cond1 = Flux2Pipeline._pack_latents(cond1_latents)
                packed_cond2 = Flux2Pipeline._pack_latents(cond2_latents)
                packed_cond_latents = torch.cat([packed_cond1, packed_cond2], dim=1)

                # Pack noisy latents and concatenate with conditions
                packed_noisy_latents = Flux2Pipeline._pack_latents(noisy_latents)
                packed_noisy_latents = torch.cat([packed_noisy_latents, packed_cond_latents], dim=1)

                # Guidance
                guidance = torch.full([1], args.guidance_scale, device=device)
                guidance = guidance.expand(latents.shape[0])

                # Forward pass
                # try:
                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                # except torch.OutOfMemoryError as e_oom:
                #     try:
                #         torch.cuda.memory._dump_snapshot("./mem_viz.pickle")
                #     except Exception as e:
                #         print(f"Failed to capture memory snapshot {e}")
                #     raise e_oom
                # else:
                #     torch.cuda.memory._record_memory_history(enabled=None)
                
                # Unpack prediction
                model_pred = model_pred[:, :packed_noisy_latents.size(1)]
                model_pred = Flux2Pipeline._unpack_latents_with_ids(model_pred, latent_ids)

                # Flow matching loss
                target = noise - latents
                loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward pass
                loss.backward()

                # Gradient clipping
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        # Clean up old checkpoints
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, checkpoint))

                        # Save checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save LoRA weights
                        transformer_lora_layers = get_peft_model_state_dict(transformer)
                        Flux2Pipeline.save_lora_weights(
                            save_directory=save_path,
                            transformer_lora_layers=transformer_lora_layers,
                        )

                progress_bar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

            if global_step >= args.max_train_steps:
                break

    # Save final LoRA weights
    transformer_lora_layers = get_peft_model_state_dict(transformer)
    Flux2Pipeline.save_lora_weights(
        save_directory=args.output_dir,
        transformer_lora_layers=transformer_lora_layers,
    )

    print(f"Training completed. LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
