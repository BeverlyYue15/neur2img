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
import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import pickle
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
#import引入 Diffusers / Transformers / Accelerate / PEFT
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from src.easycontrol_pipeline import FluxKontextPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from src.lora_helper import *
from src.layers import *
from safetensors.torch import save_file
from src.transformer_flux import FluxTransformer2DModel

if is_wandb_available():
    import wandb

from Encoder import EEGEncoder
from loongx_adapter import  DualEEGAdapter
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

def spatial_pyramid_pooling( x, output_size, adaptive=True):
        """
        Apply spatial pyramid pooling to convert variable length input to fixed length
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, variable_length]
            output_size (int): Desired fixed length after pooling
            adaptive (bool): Whether to use adaptive pooling or padding/truncation
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size, channels, output_size]
        """
        batch_size, channels, length = x.shape
        
        # If input length is already the desired length, return as is
        if length == output_size:
            return x
        
        if adaptive:
            # Calculate adaptive pool sizes to achieve the desired output size
            # We'll use a simple approach with equal-sized bins
            result = F.adaptive_avg_pool1d(x, output_size)
            
        return result
    
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux Kontext DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import FluxKontextPipeline
import torch
pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-kontextflux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)
    pipeline_args_cp = pipeline_args.copy()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        prompt = pipeline_args_cp.pop("prompt")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt, prompt_2=None)
    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                **pipeline_args_cp,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--lora_num", type=int, default=1, help="number of the lora.")
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
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="mode",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )    
    parser.add_argument(
        "--subject_column",
        type=str,
        default=None,
        help="Column in the dataset containing the fg image. Must be specified when performing I2I fine-tuning",
    )    
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="Validation image to use (during I2I fine-tuning) to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        nargs="+",
        default=[128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        nargs="+",
        default=[128],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-kontext-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--cond_size",
        type=int,
        default=512,
        help=(
            "The resolution for subject dataset"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--subject_test_images",
        type=str,
        nargs="+",
        default=["examples/subject_data/3.png"],
        help="A list of subject test image paths.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained lora path",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--use_eeg", action="store_true", help="Use EEG signal as an additional condition.")

    
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    # if args.dataset_name is not None and args.instance_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
        if args.cond_image_column is not None:
            raise ValueError("Prior preservation isn't supported with I2I training.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.cond_image_column is not None:
        assert args.image_column is not None
        # assert args.caption_column is not None
        assert args.dataset_name is not None
        assert not args.train_text_encoder
        if args.validation_prompt is not None:
            assert args.validation_image is not None and os.path.exists(args.validation_image)

    return args





def _load_folder(folder):
    """加载文件夹为 {stem: PIL.Image(RGB)}"""
    if not folder or not os.path.isdir(folder): return {}
    valid = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif','.gif','._jpg','._png'}
    out = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in valid: continue
        p = os.path.join(folder, f)
        try:
            img = Image.open(p).convert("RGB")
            out[stem] = img
        except Exception:
            pass
    return out

class EEGImageToImageDataset(Dataset):
    """
    根据 EEG + original_image 生成 transformed_image
    新增：支持全局或逐样本 mean-std 归一化
    """

    def __init__(
        self,
        packed_dict = "eeg_img_merged_train.pt",
        image_size: int = 1024,
        image_transform: Optional[transforms.Compose] = None,
        eeg_transform: Optional[transforms.Compose] = None,
        original_dir: str = "./original",
        transformed_dir: str = "./transformed",
        center_crop: bool = False,
        random_flip: bool = False,
        fallback_to_available: bool = False,
        use_caption: bool = False,
        default_prompt: str = "",
        eeg_norm: bool = True,
        eeg_norm_type: str = "global",  # "global" or "per_sample"
    ):
        # 1. 基础校验
        if isinstance(packed_dict, str):
            packed_dict = torch.load(packed_dict, map_location="cpu")
        assert "dataset" in packed_dict, "packed_dict 需包含键 'dataset'"
        self.items = packed_dict["dataset"]

        # 2. 图像相关
        self.image_transform = image_transform
        self.size = (image_size, image_size)
        if image_transform is None:
            self.image_transform_t = self._build_default_transform(
                self.size, center_crop=center_crop, random_flip=random_flip
            )
            self.image_transform_o = self._build_default_transform(
                512, center_crop=center_crop, random_flip=random_flip
            )

        # 3. EEG 归一化参数
        self.eeg_norm = eeg_norm
        self.eeg_norm_type = eeg_norm_type
        self.eeg_transform = eeg_transform
        if self.eeg_norm and self.eeg_norm_type == "global":
            self._compute_global_eeg_stats()
        self.custom_instance_prompts=None
        # 4. 图像 pkl
        self.orig: Dict[str, Any] = pickle.load(open("original.pkl", "rb"))
        self.tran: Dict[str, Any] = pickle.load(open("transformed.pkl", "rb"))
        self.fallback_to_available = fallback_to_available
        self.use_caption = use_caption
        self.default_prompt = default_prompt
        # self.description_dict=pd.read_excel('image_style_descriptions_1.xlsx')
        print(f"[INFO] loaded originals={len(self.orig)} transformed={len(self.tran)}")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _build_default_transform(self, size, center_crop=False, random_flip=True):
        ops = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)]
        ops.append(transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size))
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        return transforms.Compose(ops)

    def _compute_global_eeg_stats(self):
        """预计算全部 EEG 的均值与标准差"""
        print("[INFO] computing global EEG mean/std ...")
        # flat = torch.cat([it["eeg"].reshape(-1) for it in self.items])
        # mean, std = flat.mean(), flat.std().clamp_min(1e-8)
        # # ✅ 改为普通属性，不再用 register_buffer
        # self.eeg_mean = mean
        # self.eeg_std  = std
        self.eeg_mean = np.load("eeg_mean.npy")
        self.eeg_std  = np.load("eeg_std.npy")
        # np.save("eeg_mean.npy", self.eeg_mean)
        # np.save("eeg_std.npy", self.eeg_std)
        print(f"[INFO] EEG global mean={self.eeg_mean.item():.6f} std={self.eeg_std.item():.6f}")
        # print(f"[INFO] EEG global mean={self.eeg_mean.item():.6f} std={self.eeg_std.item():.6f}")

    def _normalize_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        if not self.eeg_norm:
            return eeg
        if self.eeg_norm_type == "global":
            # ✅ 运行时保护：如果忘记计算全局统计量，就当场算一遍
            if not hasattr(self, "eeg_mean"):
                self._compute_global_eeg_stats()
            return (eeg - self.eeg_mean) / self.eeg_std
        else:  # per_sample
            # mean_, std_ = eeg.mean(), eeg.std().clamp_min(1e-8)
            # return (eeg - mean_) / std_
            return eeg

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def _pair(self, stem: str):
        img_o = self.orig.get(stem, None)
        img_t = self.tran.get(stem, None)
        if self.fallback_to_available:
            if img_o is None and img_t is not None:
                img_o = img_t
            if img_t is None and img_o is not None:
                img_t = img_o
        return img_o, img_t

    def __getitem__(self, idx):
        item = self.items[idx]
        eeg = item["eeg"]                      # Tensor[C,T,...]
        eeg = self._normalize_eeg(eeg)         # 全局/逐样本归一化
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)

        stem = str(item.get("name", idx))
        img_o, img_t = self._pair(stem)
        if img_o is None and img_t is None:
            raise FileNotFoundError(f"Image '{stem}' not found in original/transformed.")

        if img_o is not None:
            img_o = self.image_transform_o(img_o)
        if img_t is not None:
            img_t = self.image_transform_t(img_t)

        prompt = item.get("caption", None) if self.use_caption else None
        if prompt is None:
            prompt = self.default_prompt

        # NaN/Inf 保护
        def _check(t, name):
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    raise RuntimeError(f"[NaNGuard:Dataset] {name} has NaN/Inf")

        _check(eeg, "eeg")
        _check(img_o if img_o is not None else torch.tensor(0.0), "img_o")
        _check(img_t if img_t is not None else torch.tensor(0.0), "img_t")

        return {
            "eeg_values": eeg,
            "pixel_values": img_t,
            "cond_pixel_values": img_o,
            "prompts": prompt,
            "label": stem,
        }

def collate_fn_eeg(examples, with_prior_preservation=False):
    # 目标图
    pixel_values = [ex["pixel_values"] for ex in examples if ex["pixel_values"] is not None]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

    # 条件图
    cond_pixel_values = [ex["cond_pixel_values"] for ex in examples if ex["cond_pixel_values"] is not None]
    cond_pixel_values = torch.stack(cond_pixel_values).to(memory_format=torch.contiguous_format).float()

    # EEG
    eeg_values = [ex["eeg_values"] for ex in examples]
    eeg_values = torch.stack(eeg_values).float()

    prompts = [ex["prompts"] for ex in examples]

    batch = {
        "pixel_values": pixel_values,
        "cond_pixel_values": cond_pixel_values,
        "eeg_values": eeg_values,
        "prompts": prompts,
    }

    if with_prior_preservation:
        # 如果你还想混入 class images（不常见于 i2i 场景，可按需启用）
        raise NotImplementedError("Prior preservation is uncommon for EEG+i2i; add if needed.")

    return batch

def eeg_adapter_decode(
    eeg,
    adapter
):
    
    prompt_embeds = adapter(eeg, output_type="seq")
    
    pooled_prompt_embeds = adapter(eeg, output_type="global")
    
    text_ids = torch.zeros(prompt_embeds.shape[1], 3)
    
    return prompt_embeds, pooled_prompt_embeds, text_ids




def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16

            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            del pipeline
            free_memory()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    eegencoder=EEGEncoder(device=accelerator.device,dtype=torch.float32)
    eeg_adapter=DualEEGAdapter()
    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    eegencoder.requires_grad_(True)
    eeg_adapter.requires_grad_(True)
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    eegencoder.to(accelerator.device)
    eeg_adapter.to(accelerator.device)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    #### lora_layers ####
    if args.pretrained_lora_path is not None:
        lora_path = args.pretrained_lora_path
        checkpoint = load_checkpoint(lora_path)
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        number = 1
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.rank, network_alphas=args.lora_alpha, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                print("setting LoRA Processor for", name)        
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.rank, network_alphas=args.lora_alpha, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()
    else:
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.rank, network_alphas=args.lora_alpha, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=args.rank, network_alphas=args.lora_alpha, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
            else:
                lora_attn_procs[name] = attn_processor        
    ######################
    transformer.set_attn_processor(lora_attn_procs)
    transformer.train()
    for n, param in transformer.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        if args.use_eeg: 
            models.append(eegencoder)
            models.append(eeg_adapter)
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
    
    opt_groups = [{"params": [p for p in transformer.parameters() if p.requires_grad], "lr": args.learning_rate}]
    if args.use_eeg:
        opt_groups.append({"params": eegencoder.parameters(), "lr": args.learning_rate})  # 或自定义
        opt_groups.append({"params": eeg_adapter.parameters(), "lr": args.learning_rate})  # 或自定义

        
    optimizer = torch.optim.AdamW(opt_groups, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.adam_weight_decay)

    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')

    # optimizer_class = torch.optim.AdamW
    # optimizer = optimizer_class(
    #     [transformer_parameters_with_lr],
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    
    
    
    
    
    
    
    
    
    
    # # Dataset and DataLoaders creation:
    # train_dataset = DreamBoothDataset(
    #     instance_data_root=args.instance_data_dir,
    #     instance_prompt=args.instance_prompt,
    #     class_prompt=args.class_prompt,
    #     class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    #     class_num=args.num_class_images,
    #     buckets=buckets,
    #     repeats=args.repeats,
    #     center_crop=args.center_crop,
    #     args=args,
    # )
    # if args.cond_image_column is not None:
    #     logger.info("I2I fine-tuning enabled.")
    # batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=False)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_sampler=batch_sampler,
    #     collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    #     num_workers=args.dataloader_num_workers,
    # )
    train_dataset = EEGImageToImageDataset(
        packed_dict=torch.load("./1108_04valstyle/train_multi.pt"),  # 或者你自行先 load 再传
        image_size=512,
        eeg_transform=None,                      # 若需要可传
        original_dir="./original",
        transformed_dir="./transformed",
        center_crop=False,
        random_flip=False,
        fallback_to_available=False,
        use_caption=False,
        default_prompt=args.instance_prompt,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda exs: collate_fn_eeg(exs, with_prior_preservation=False),
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        text_encoder_one.cpu(), text_encoder_two.cpu()
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

#     if not train_dataset.custom_instance_prompts:
#         if not args.train_text_encoder:
#             prompt_embeds = instance_prompt_hidden_states
#             pooled_prompt_embeds = instance_pooled_prompt_embeds
#             text_ids = instance_text_ids
#             if args.with_prior_preservation:
#                 prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
#                 pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
#                 text_ids = torch.cat([text_ids, class_text_ids], dim=0)
#         # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
#         # we need to tokenize and encode the batch prompts on all training steps
#         else:
#             tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
#             tokens_two = tokenize_prompt(
#                 tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
#             )
#             if args.with_prior_preservation:
#                 class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, max_sequence_length=77)
#                 class_tokens_two = tokenize_prompt(
#                     tokenizer_two, args.class_prompt, max_sequence_length=args.max_sequence_length
#                 )
#                 tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
#                 tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

#     elif train_dataset.custom_instance_prompts and not args.train_text_encoder:
#         cached_text_embeddings = []
#         for batch in tqdm(train_dataloader, desc="Embedding prompts"):
#             batch_prompts = batch["prompts"]
#             prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
#                 batch_prompts, text_encoders, tokenizers
#             )
#             cached_text_embeddings.append((prompt_embeds, pooled_prompt_embeds, text_ids))

#         if args.validation_prompt is None:
#             text_encoder_one.cpu(), text_encoder_two.cpu()
#             del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
#             free_memory()

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    # has_image_input = args.cond_image_column is not None
    has_image_input=True
    if args.cache_latents:
        latents_cache = []
        cond_latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if has_image_input:
                    batch["cond_pixel_values"] = batch["cond_pixel_values"].to(
                        accelerator.device, non_blocking=True, dtype=weight_dtype
                    )
                    cond_latents_cache.append(vae.encode(batch["cond_pixel_values"]).latent_dist)
                    
        if args.validation_prompt is None:
            vae.cpu()
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, eegencoder, eeg_adapter,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, eegencoder, eeg_adapter,optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "easy_control-kontext_based-code"
        tracker_config = dict(vars(args))
        tracker_config.pop('rank')
        tracker_config.pop('lora_alpha')
        tracker_config.pop('subject_test_images')
        accelerator.init_trackers(tracker_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    has_guidance = unwrap_model(transformer).config.guidance_embeds
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            if args.use_eeg: 
                models_to_accumulate.append(eegencoder)
                models_to_accumulate.append(eeg_adapter)
                
            with accelerator.accumulate(models_to_accumulate):
#                 prompts = batch["prompts"]

#                 # encode batch prompts when custom prompts are provided for each image -
#                 if train_dataset.custom_instance_prompts:
#                     if not args.train_text_encoder:
#                         prompt_embeds, pooled_prompt_embeds, text_ids = cached_text_embeddings[step]
#                     else:
#                         tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
#                         tokens_two = tokenize_prompt(
#                             tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
#                         )
#                         prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
#                             text_encoders=[text_encoder_one, text_encoder_two],
#                             tokenizers=[None, None],
#                             text_input_ids_list=[tokens_one, tokens_two],
#                             max_sequence_length=args.max_sequence_length,
#                             device=accelerator.device,
#                             prompt=prompts,
#                         )
#                 else:
#                     elems_to_repeat = len(prompts)
#                     if args.train_text_encoder:
#                         prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
#                             text_encoders=[text_encoder_one, text_encoder_two],
#                             tokenizers=[None, None],
#                             text_input_ids_list=[
#                                 tokens_one.repeat(elems_to_repeat, 1),
#                                 tokens_two.repeat(elems_to_repeat, 1),
#                             ],
#                             max_sequence_length=args.max_sequence_length,
#                             device=accelerator.device,
#                             prompt=args.instance_prompt,
#                         )

                # Convert images to latent space
                if args.cache_latents:
                    if args.vae_encode_mode == "sample":
                        model_input = latents_cache[step].sample()
                        if has_image_input:
                            cond_model_input = cond_latents_cache[step].sample()
                    else:
                        model_input = latents_cache[step].mode()
                        if has_image_input:
                            cond_model_input = cond_latents_cache[step].mode()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    if has_image_input:
                        cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)
                    if args.vae_encode_mode == "sample":
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    else:
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                if has_image_input:
                    cond_model_input = (cond_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                    cond_model_input = cond_model_input.to(dtype=weight_dtype)
                    
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                offset = 64
                
                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                if has_image_input:
                    cond_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        cond_model_input.shape[0],
                        cond_model_input.shape[2] // 2,
                        cond_model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    cond_latents_ids[..., 0] = 1
                    latent_image_ids = torch.cat([latent_image_ids, cond_latents_ids], dim=0)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape
                if has_image_input:
                    packed_cond_input = FluxKontextPipeline._pack_latents(
                        cond_model_input,
                        batch_size=cond_model_input.shape[0],
                        num_channels_latents=cond_model_input.shape[1],
                        height=cond_model_input.shape[2],
                        width=cond_model_input.shape[3],
                    )
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_input], dim=1)
                    
                latent_image_ids_to_concat = [latent_image_ids]
                packed_cond_model_input_to_concat = []
#                 if args.subject_column is not None:
#                     subject_pixel_values = batch["subject_pixel_values"].to(dtype=vae.dtype)
#                     subject_input = vae.encode(subject_pixel_values).latent_dist.mode()
#                     subject_input = (subject_input - vae_config_shift_factor) * vae_config_scaling_factor
#                     subject_input = subject_input.to(dtype=weight_dtype)             
#                     sub_number = subject_pixel_values.shape[-2] // args.cond_size
#                     latent_subject_ids = FluxKontextPipeline._prepare_latent_image_ids(
#                         subject_input.shape[0],
#                         subject_input.shape[2] // 2,
#                         subject_input.shape[3] // 2,
#                         accelerator.device,
#                         weight_dtype,
#                     )
#                     latent_subject_ids[:, 1] += offset
#                     sub_latent_image_ids = torch.concat([latent_subject_ids for _ in range(sub_number)], dim=-2)
#                     latent_image_ids_to_concat.append(sub_latent_image_ids)
                    
#                     packed_subject_model_input = FluxKontextPipeline._pack_latents(    
#                         subject_input,
#                         batch_size=subject_input.shape[0],
#                         num_channels_latents=subject_input.shape[1],
#                         height=subject_input.shape[2],
#                         width=subject_input.shape[3],
#                     )
#                     packed_cond_model_input_to_concat.append(packed_subject_model_input)   
                    
                if args.use_eeg:
                    subject_pixel_values = batch["eeg_values"].to(dtype=eegencoder.module.dtype,device=eegencoder.module.device)
                    # print('EEG dtype before encoder:', subject_pixel_values.dtype)
                    # print(subject_pixel_values.device)
                    # print(eegencoder.device)
                    # subject_input = vae.encode(subject_pixel_values).latent_dist.mode()
                    # subject_input = (subject_input - vae_config_shift_factor) * vae_config_scaling_factor
                    # subject_input = subject_input.to(dtype=weight_dtype)
                    subject_input=eegencoder(subject_pixel_values).to(dtype=weight_dtype) 
                    prompt_embeds, pooled_prompt_embeds, text_ids=eeg_adapter_decode(subject_input,eeg_adapter)
                    prompt_embeds=prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                    pooled_prompt_embeds=pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                    text_ids=text_ids.to(accelerator.device, dtype=weight_dtype)
                    assert torch.isfinite(subject_input).all(), "[NaNGuard] EEGEncoder output has NaN/Inf"
                    
                    # print("final")
                    # print(subject_input.shape)
                    # sub_number = subject_pixel_values.shape[-2] // args.cond_size
                    sub_number=1
                    latent_subject_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        subject_input.shape[0],
                        subject_input.shape[2] // 2,
                        subject_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    latent_subject_ids[:, 1] += offset
                    sub_latent_image_ids = torch.concat([latent_subject_ids for _ in range(sub_number)], dim=-2)
                    latent_image_ids_to_concat.append(sub_latent_image_ids)
                    
                    packed_subject_model_input = FluxKontextPipeline._pack_latents(    
                        subject_input,
                        batch_size=subject_input.shape[0],
                        num_channels_latents=subject_input.shape[1],
                        height=subject_input.shape[2],
                        width=subject_input.shape[3],
                    )
                    packed_cond_model_input_to_concat.append(packed_subject_model_input)   
                latent_image_ids = torch.concat(latent_image_ids_to_concat, dim=-2)
                cond_packed_noisy_model_input = torch.concat(packed_cond_model_input_to_concat, dim=-2)
                
                # Kontext always has guidance
                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    cond_hidden_states=cond_packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                if has_image_input:
                    model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # params_to_clip = (
                    #     itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                    #     if args.train_text_encoder
                    #     else transformer.parameters()
                    # )
                    # accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    params_to_clip = itertools.chain(
                        transformer.parameters(),
                        (text_encoder_one.parameters() if args.train_text_encoder else []),
                        (eegencoder.parameters() if args.use_eeg else []),
                        (eeg_adapter.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
                        lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        save_folder=os.path.join(save_path,f"eeg_encoder_weights.pt")
                        torch.save(eegencoder.state_dict(),save_folder)
                        save_folder=os.path.join(save_path,f"eeg_adapter_weights.pt")
                        torch.save(eeg_adapter.state_dict(),save_folder)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

#             if accelerator.is_main_process and accelerator.sync_gradients:
#                 if args.validation_prompt is not None and global_step % args.validation_steps == 0:
#                     # create pipeline
#                     if not args.train_text_encoder:
#                         text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
#                         text_encoder_one.to(weight_dtype)
#                         text_encoder_two.to(weight_dtype)
#                     pipeline = FluxKontextPipeline.from_pretrained(
#                         args.pretrained_model_name_or_path,
#                         vae=vae,
#                         text_encoder=unwrap_model(text_encoder_one),
#                         text_encoder_2=unwrap_model(text_encoder_two),
#                         transformer=unwrap_model(transformer),
#                         revision=args.revision,
#                         variant=args.variant,
#                         torch_dtype=weight_dtype,
#                     )
                    
#                     ######## 
#                     if len(args.subject_test_images) != 0 and args.subject_test_images != ['None']:
#                         subject_paths = args.subject_test_images
#                         subject_ls = [Image.open(image_path).convert("RGB") for image_path in subject_paths]
#                     else:
#                         subject_ls = []
#                     ########

#                     pipeline_args = {"prompt": args.validation_prompt,
#                                     "subject_images": subject_ls,
#                                     "cond_size": args.cond_size
#                                     }
#                     if has_image_input and args.validation_image:
#                         pipeline_args.update({"image": load_image(args.validation_image)})
#                     # images = log_validation(
#                     #     pipeline=pipeline,
#                     #     args=args,
#                     #     accelerator=accelerator,
#                     #     pipeline_args=pipeline_args,
#                     #     epoch=epoch,
#                     #     torch_dtype=weight_dtype,
#                     # )
#                     save_path = os.path.join(args.output_dir, "validation")
#                     os.makedirs(save_path, exist_ok=True)
#                     save_folder = os.path.join(save_path, f"checkpoint-{global_step}")
#                     os.makedirs(save_folder, exist_ok=True)
#                     # for idx, img in enumerate(images):
#                     #     img.save(os.path.join(save_folder, f"{idx}.jpg"))
#                     del text_encoder_one, text_encoder_two
#                     free_memory()

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)