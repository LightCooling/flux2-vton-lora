# Repository Notes

Updated: 2026-04-23

## Purpose

This repository contains compact FLUX.2 LoRA experiments for virtual try-on domain adaptation under tight VRAM constraints.

The goal is to keep the training path small and reproducible:

- text encoder outputs are precomputed into prompt embedding caches;
- VAE outputs are precomputed into latent caches;
- `train.py` fine-tunes only the FLUX.2 transformer LoRA weights;
- `generate_i2i.py` reflects the intended VTON inference workflow.

## Design Constraints

- No validation pipeline is built into `train.py`.
- Prompt embeddings default to `128` tokens.
- The repository prioritizes lower VRAM usage over a full trainer feature set.

## Cache Contract

`prepare_cache.py` writes five tensor families:

- `prompt_embeds_*.pt`
- `text_ids_*.pt`
- `latents_*.pt`
- `cond1_latents_*.pt`
- `cond2_latents_*.pt`

`train.py` expects:

- contiguous indices starting at `000000`;
- the same index set for all five prefixes;
- matching latent shapes across target and condition tensors.

During training, target and condition tokens are concatenated for transformer input, but the loss is computed only on target latents.

## Publishing Guidelines

Commit source files and documentation only.

Do not commit:

- `data/`
- `cache/`
- `output/`
- `.venv/`
- notebooks
- local experiment scripts or memory snapshots

`uv.lock` remains ignored. The `diffusers` dependency is pinned in `pyproject.toml` to the revision used by this project.

## Quick Checks

```bash
uv run python -m py_compile prepare_cache.py train.py generate_i2i.py generate_t2i.py
uv run python -c "from train import SimpleFluxDataset; ds = SimpleFluxDataset('cache'); print(len(ds))"
git status --short
```

