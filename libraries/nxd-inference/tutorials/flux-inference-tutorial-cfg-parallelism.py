# %% [markdown]
# # Generating Images with Black Forest Labs Flux.1-Dev on Trn1/Trn2 (CFG Parallelism)
#
# This tutorial provides a step-by-step guide for generating images using the Flux.1-dev model from Black Forest Labs with NeuronX Distributed (NxD) Inference on a single trn2.48xl instance, using **CFG (Classifier-Free Guidance) Parallelism**.
#
# CFG Parallelism batches the negative and positive prompt inference together and distributes them across data-parallel ranks,
# reducing the number of transformer forward passes from 2 to 1 per denoising step.
# This is mutually exclusive with Context Parallelism.

# %%
# .. contents:: Table of contents
#     :local:
#     :depth: 2

# %% [markdown]
# ## Background, Concepts, and Optimizations
#
# ### CFG Parallelism
#
# When using True Classifier-Free Guidance (CFG), the model must run inference on both a negative prompt and a positive prompt at each denoising step.
# Without CFG Parallelism, this requires two sequential transformer forward passes per step.
# With CFG Parallelism enabled, the negative and positive inputs are batched together and scattered across data-parallel ranks (dim=0),
# so each rank processes one of the two prompts in parallel. The outputs are then gathered and the CFG formula is applied.
# This effectively halves the transformer computation per denoising step.
#
# The `world_size` is set to `2 × backbone_tp_degree` to provide 2 data-parallel ranks.
#

# %% [markdown]
# ## Step 1: Setup the environment
# ### Set up and connect to a trn2.48xlarge instance
#
# As a prerequisite, this tutorial requires that you have a Trn2 instance created from a Deep Learning AMI that has the Neuron SDK pre-installed.
# To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK, see the [NxDI setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html#nxdi-setup).
#
# After setting up an instance, use SSH to connect to the Trn2 instance using the key pair that you chose when you launched the instance.
#
# To use Jupyter Notebook on the Neuron instance, follow the [Jupyter Notebook QuickStart guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/notebook/setup-jupyter-notebook-steps-troubleshooting.html).
#
# After you are connected, activate the Python virtual environment that includes the Neuron SDK.
#
# `source ~/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
#
# Run pip list to verify that the Neuron SDK is installed.
#
# `pip list | grep neuron`
#
# You should see Neuron packages including neuronx-distributed-inference and neuronx-cc.
#
# ### Download the model
#
# To use this sample, you must first download the model checkpoint from HuggingFace to a local path on the Trn2 instance. For more information, see [Download models](https://huggingface.co/docs/hub/en/models-downloading) in the HuggingFace documentation. You can download and use [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) for this tutorial.

# %%
# !pip install matplotlib

# %%
import os
import torch
from matplotlib import pyplot as plt

from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)


# %% [markdown]
# ## Step 2: Setup Inference Parameters and Model Config
#
# Start by initializing your inference parameters, which include model parallelism configuration, image sizes and model configuration. Ensure that `CKPT_DIR` matches the local directory where you downloaded the model in Step 1.
#
# For CFG Parallelism, the `world_size` is automatically calculated as `2 × backbone_tp_degree`.
# With `backbone_tp_degree=4`, this gives `world_size=8`, providing 2 data-parallel ranks for batched CFG inference.

# %%
# !hf download black-forest-labs/FLUX.1-dev --local-dir /opt/dlami/nvme/models/FLUX.1-dev

# %%
backbone_tp_degree = 4
dtype = torch.bfloat16

height, width = [1024, 1024]
guidance_scale = 3.5
num_inference_steps = 25
prompt = "A robot named trn2"
negative_prompt = "ugly, distorted, low quality"
true_cfg_scale = 2.0

# The Ckpt directory root under huggingface
CKPT_DIR = "/opt/dlami/nvme/models/FLUX.1-dev/"

# Existing Compiled working directory for the compiler
BASE_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir_cfg_parallelism/"

# Calculate world_size with CFG Parallelism (2 × backbone_tp_degree)
world_size = get_flux_parallelism_config(
    backbone_tp_degree,
    cfg_parallel_enabled=True,
)
print(f"backbone_tp_degree={backbone_tp_degree}, world_size={world_size}")

# %% [markdown]
# ## Step 3: Setup Model and Neuron Configuration
#
# Here, you use the `create_flux_config` helper to initialize configuration objects for all component models in the Flux Pipeline (CLIP, T5, backbone transformer, and VAE).
# The `cfg_parallel_enabled=True` flag configures the backbone transformer for CFG Parallelism.
#
# Parallelism configuration:
# - CLIP: `tp_degree` of 1
# - T5: `tp_degree` equals `world_size` (8 in this example)
# - Backbone transformer: `tp_degree` equals `backbone_tp_degree` (4), with 2 data-parallel ranks for CFG
# - VAE: `tp_degree` of 1

# %%
clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
    CKPT_DIR,
    world_size,
    backbone_tp_degree,
    dtype,
    height,
    width,
    cfg_parallel_enabled=True,
)

# %% [markdown]
# ## Step 4: Initialize the Flux Application and Compile
#
# Now you instantiate the `NeuronFluxApplication` which contains the pipeline orchestration logic, as well as the various component models. You then compile the application, which compiles each component model individually.

# %%
flux_app = NeuronFluxApplication(
    model_path=CKPT_DIR,
    text_encoder_config=clip_config,
    text_encoder2_config=t5_config,
    backbone_config=backbone_config,
    decoder_config=decoder_config,
    height=height,
    width=width,
)

flux_app.compile(BASE_COMPILE_WORK_DIR)

# %% [markdown]
# ## Step 5: Load Model
# This step loads the compiled model (NEFF), along with the model weights into device memory. Specifically, calling load on the flux_app loads all the individual component models.

# %%
flux_app.load(BASE_COMPILE_WORK_DIR)

# %% [markdown]
# ## Step 6: Generate an Image
#
# Generate images using CFG Parallelism by providing a `negative_prompt` and `true_cfg_scale`.
# The pipeline automatically batches the negative and positive prompts for parallel inference.

# %%
for _ in range(5):
    image = flux_app(
        prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    # plt.imshow(image)
    # plt.show()

import time
start_time = time.time()
image = flux_app(
    prompt,
    negative_prompt=negative_prompt,
    true_cfg_scale=true_cfg_scale,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
).images[0]
time_25 = time.time() - start_time

print('time for 25 steps:', time_25)

# %% [markdown]
# ## Notes
# ### CFG Parallelism vs Context Parallelism
#
# CFG Parallelism and Context Parallelism are **mutually exclusive**.
# - **CFG Parallelism** splits the batch dimension (negative vs positive prompts) across data-parallel ranks.
# - **Context Parallelism** splits the sequence dimension across data-parallel ranks.
#
# Both require `world_size = 2 × backbone_tp_degree`.
#
# ### Running on trn1
#
# This sample can also be deployed to a trn1.32xlarge by setting:
#
# ```
# backbone_tp_degree = 8
# ```
#
# The `world_size` will be automatically calculated as `2 × 8 = 16`.
