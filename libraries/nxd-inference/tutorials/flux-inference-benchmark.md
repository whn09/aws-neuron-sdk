# Flux.1-Dev Inference Benchmark: CFG Parallelism on Trn2

## Overview

This document summarizes the benchmark results for running [Black Forest Labs Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) image generation on a **trn2.48xlarge** instance using NeuronX Distributed (NxD) Inference. We compare the baseline (Context Parallelism with serial True CFG) against the new **CFG Parallelism** mode.

## What is CFG Parallelism?

Classifier-Free Guidance (CFG) generates higher-quality images by running inference on both a negative prompt and a positive prompt, then combining the results. Without parallelism, this requires **two sequential** transformer forward passes per denoising step, roughly doubling latency.

**CFG Parallelism** batches the negative and positive prompt inputs together and distributes them across 2 data-parallel ranks (scatter on batch dimension). Each rank processes one prompt in parallel, and the outputs are gathered to apply the CFG formula. This effectively reduces the transformer calls from 2 to 1 per step.

CFG Parallelism and Context Parallelism are **mutually exclusive** — both require `world_size = 2 × backbone_tp_degree`.

## Environment

| Item | Value |
|------|-------|
| Instance | trn2.48xlarge |
| Neuron SDK | neuronx-cc 2.23, neuronx-distributed-inference 0.9.0 |
| PyTorch | 2.9.0 |
| Image Resolution | 1024 × 1024 |
| Inference Steps | 25 |
| backbone_tp_degree | 4 |
| world_size | 8 |

## Benchmark Results

| Mode | True CFG | Throughput (it/s) | 25-step Latency (s) |
|------|----------|-------------------|----------------------|
| No True CFG (`true_cfg_scale=1.0`) | No | 5.85 | 4.65 |
| Baseline (Context Parallel, serial True CFG) | Yes (`true_cfg_scale=2.0`) | 2.93 | 8.90 |
| **CFG Parallelism** | Yes (`true_cfg_scale=2.0`) | **3.69** | **7.15** |

## Key Takeaways

- Enabling True CFG without parallelism roughly **doubles** the per-step latency (from 4.65s to 8.90s) due to two sequential transformer forward passes.
- **CFG Parallelism recovers ~20% of that overhead** (8.90s → 7.15s) by distributing the negative/positive prompt inference across 2 data-parallel ranks in a single batched forward pass.
- Compared to no True CFG (4.65s), CFG Parallelism adds only ~54% overhead while providing the full quality benefits of classifier-free guidance.

## Scripts

| Script | Description |
|--------|-------------|
| `flux-inference-tutorial.py` | Original tutorial with True CFG support (negative_prompt + true_cfg_scale) |
| `flux-inference-tutorial-cfg-parallelism.py` | CFG Parallelism tutorial using `cfg_parallel_enabled=True` and helper functions |

## Configuration Differences

### Baseline (flux-inference-tutorial.py)

```python
world_size = 8
backbone_tp_degree = 4

# Manual config setup, no cfg_parallel_enabled
backbone_config = FluxBackboneInferenceConfig(
    neuron_config=backbone_neuron_config,
    load_config=load_diffusers_config(backbone_path),
    height=height, width=width,
)

# Pipeline call with True CFG (serial, 2 passes per step)
image = flux_app(
    prompt,
    negative_prompt=negative_prompt,
    true_cfg_scale=2.0,
    ...
).images[0]
```

### CFG Parallelism (flux-inference-tutorial-cfg-parallelism.py)

```python
backbone_tp_degree = 4

# Auto world_size calculation: 2 × 4 = 8
world_size = get_flux_parallelism_config(
    backbone_tp_degree, cfg_parallel_enabled=True
)

# Helper function with cfg_parallel_enabled
clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
    CKPT_DIR, world_size, backbone_tp_degree, dtype,
    height, width, cfg_parallel_enabled=True,
)

# Pipeline call with True CFG (parallel, 1 pass per step)
image = flux_app(
    prompt,
    negative_prompt=negative_prompt,
    true_cfg_scale=2.0,
    ...
).images[0]
```
