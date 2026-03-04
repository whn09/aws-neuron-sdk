# %% [markdown]
# # Generating Images with Black Forest Labs Flux.1-Dev on Trn1/Trn2
# 
# This tutorial provides a step-by-step guide for generating images using the Flux.1-dev model from Black Forest Labs with NeuronX Distributed (NxD) Inference on a single trn2.48xl instance. This sample specifically generates 1k x 1k images.

# %%
# .. contents:: Table of contents
#     :local:
#     :depth: 2

# %% [markdown]
# ## Background, Concepts, and Optimizations
# 
# ### Tensor and Context Parallel
# 
# For the latent transformer model, use a combination of Tensor Parallelism and Context Parallelism. Due to the compute-bound nature of diffusion inference, add additional parallelism by using sharding on the sequence dimension. Sharding is governed by the `world_size` relative to the `backbone_tp_degree`. 
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


from neuronx_distributed_inference.models.diffusers.flux.application import NeuronFluxApplication
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import CLIPInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import T5InferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import FluxBackboneInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import VAEDecoderInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config


# %% [markdown]
# ## Step 2: Setup Inference Parameters and Model Config
# 
# Start by initializing your inference parameters, which include model parallelism configuration, image sizes and model configuration. Ensure that that `CKPT_DIR` matches the local directory where you downloaded the model in Step 1.

# %%
# !hf download black-forest-labs/FLUX.1-dev --local-dir /opt/dlami/nvme/models/FLUX.1-dev

# %%
world_size = 8
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
BASE_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir/"


# %% [markdown]
# ## Step 3: Setup Model and Neuron Configuration
# 
# Here, you initialize the various component model configuration objects for the models within the Flux Pipeline. The Flux pipeline contains CLIP, T5, the backbone transformer and the VAE. For each component model, you can use the following parallelism configuration:
# - For CLIP, `tp_degree` of 1
# - For T5, `tp_degree` is the same as the `world_size`. In the case of this example, this will be 8.
# - For the backbone transformer, if using Context Parallelism, `tp_degree` is half the world size. In the case of this example, this will be 4, which allows for 2 CP ranks.
# - Finally, for the VAE, `tp_degree` of 1.

# %%
text_encoder_path = os.path.join(CKPT_DIR, "text_encoder")
text_encoder_2_path = os.path.join(CKPT_DIR, "text_encoder_2")
backbone_path = os.path.join(CKPT_DIR, "transformer")
vae_decoder_path = os.path.join(CKPT_DIR, "vae")

clip_neuron_config = NeuronConfig(
    tp_degree=1,
    world_size=world_size,
    torch_dtype=dtype,
)
clip_config = CLIPInferenceConfig(
    neuron_config=clip_neuron_config,
    load_config=load_pretrained_config(text_encoder_path),
)

t5_neuron_config = NeuronConfig(
    tp_degree=world_size,
    world_size=world_size,
    torch_dtype=dtype,
)
t5_config = T5InferenceConfig(
    neuron_config=t5_neuron_config,
    load_config=load_pretrained_config(text_encoder_2_path),
)

backbone_neuron_config = NeuronConfig(
    tp_degree=backbone_tp_degree,
    world_size=world_size,
    torch_type=dtype,
)
backbone_config = FluxBackboneInferenceConfig(
    neuron_config=backbone_neuron_config,
    load_config=load_diffusers_config(backbone_path),
    height=height,
    width=width,
)

decoder_neuron_config = NeuronConfig(
    tp_degree=1,
    world_size=world_size,
    torch_type=dtype,
)
decoder_config = VAEDecoderInferenceConfig(
    neuron_config=decoder_neuron_config,
    load_config=load_diffusers_config(vae_decoder_path),
    height=height,
    width=width,
    transformer_in_channels=backbone_config.in_channels,
)

setattr(
    backbone_config,
    "vae_scale_factor",
    decoder_config.vae_scale_factor,
)

# %% [markdown]
# ## Step 4: Initialize the Flux Application and Compile
# 
# Now you instantiate the `NeuronFluxApplication` which contains the pipeline orchestration logic, as well as the various component models. You then compile the application, which then compiles each component model individually.

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
# Finally, you will generate a singular image and render it:

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
# ### Running Flux Inference on trn1
# 
# This sample can also be deployed to a trn1.32xlarge with a few modifications. If you are using Context Parallelism specifically, then apply the following parallelism configuration
# 
# ```
# world_size = 16
# backbone_tp_degree = 8
# ```
# 
# Otherwise use the following:
# 
# ```
# world_size = 8
# backbone_tp_degree = 8
# ```


