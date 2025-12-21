# EEG-Guided Style Transfer Training with Flux Kontext

This repository contains training code for EEG-guided style transfer using the Flux Kontext model with LoRA adapters.

## Overview

This project implements a novel approach to style transfer by using EEG (electroencephalography) signals as conditioning input to guide the style transformation process. The model is based on FLUX.1-Kontext-dev and uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

### Key Features

- **EEG Signal Integration**: Uses brain signals to condition the style transfer process
- **Multi-Modal Architecture**: Combines EEG encoder, dual EEG adapter, and Flux transformer
- **LoRA Fine-tuning**: Efficient parameter-efficient training approach
- **Flexible Resolution**: Supports variable input resolutions with aspect ratio buckets
- **Distributed Training**: Built on Accelerate for multi-GPU support

## Architecture Components

### Main Models

1. **FluxTransformer2DModel**: Base diffusion transformer from FLUX.1-Kontext-dev
2. **EEGEncoder**: Custom encoder for processing EEG signals
3. **DualEEGAdapter**: Adapter that converts EEG features to text embeddings and pooled embeddings
4. **VAE (AutoencoderKL)**: Encodes/decodes images to/from latent space
5. **Text Encoders**: CLIP and T5 for text conditioning (frozen during training)

### LoRA Architecture

- **Double Stream Blocks**: 19 transformer blocks with LoRA on Q, K, V, and projection layers
- **Single Stream Blocks**: 38 transformer blocks with LoRA on Q, K, V layers
- **Rank**: 128 (configurable)
- **Alpha**: 128 (configurable)

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers >= 4.30.0
- Diffusers >= 0.34.0
- Accelerate >= 0.20.0
- PEFT >= 0.4.0
- CUDA-capable GPU (recommended)

## Dataset Structure

The training script expects an EEG-Image paired dataset with the following structure:

### Dataset Format

The code uses a custom `EEGImageToImageDataset` that loads:

1. **Packed Dictionary** (`train_multi.pt`): Contains EEG signals and metadata
   - Structure: `{"dataset": [{"eeg": tensor, "name": str, "caption": str}, ...]}`

2. **Image Files**:
   - `original.pkl`: Original images (before style transfer)
   - `transformed.pkl`: Target styled images (after style transfer)

3. **EEG Statistics** (for normalization):
   - `eeg_mean.npy`: Global mean of EEG signals
   - `eeg_std.npy`: Global standard deviation of EEG signals

### Data Loading

```python
train_dataset = EEGImageToImageDataset(
    packed_dict=torch.load("./1108_04valstyle/train_multi.pt"),
    image_size=512,
    original_dir="./original",
    transformed_dir="./transformed",
    center_crop=False,
    random_flip=False,
    default_prompt="Using the provided EEG signals to perform style transfer."
)
```

## Training Configuration

### Accelerate Configuration

Create a file named `kontext_easycontrol.yaml` for Accelerate:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU  # or NO for single GPU
mixed_precision: bf16
num_processes: 4  # number of GPUs
gpu_ids: all
```

### Training Script

Run the training using the provided shell script:

```bash
bash train_style.sh
```

### Training Parameters

#### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--pretrained_model_name_or_path` | `black-forest-labs/FLUX.1-Kontext-dev` | Base model from HuggingFace |
| `--rank` | 128 | LoRA rank dimension |
| `--lora_alpha` | 128 | LoRA scaling factor |
| `--use_eeg` | flag | Enable EEG conditioning |

#### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--optimizer` | adamw | Optimizer type |
| `--lr_scheduler` | constant | Learning rate schedule |
| `--lr_warmup_steps` | 0 | Number of warmup steps |
| `--max_train_steps` | 12000 | Total training steps |
| `--train_batch_size` | 1 | Batch size per device |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--mixed_precision` | bf16 | Mixed precision training |

#### Image Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--resolution` | 512 | Target image resolution |
| `--cond_size` | 512 | Conditioning image size |
| `--guidance_scale` | 3.5 | Guidance scale for diffusion |

#### Checkpointing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--output_dir` | `./style_adapter_eeg_1-80_multi` | Output directory |
| `--checkpointing_steps` | 500 | Save checkpoint every N steps |
| `--checkpoints_total_limit` | 10 | Maximum checkpoints to keep |
| `--validation_steps` | 2000 | Run validation every N steps |
| `--num_validation_images` | 2 | Number of validation images |

#### Other Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--seed` | 0 | Random seed for reproducibility |
| `--instance_prompt` | "Using the provided EEG signals to perform style transfer." | Default prompt |

## Training Process

### Workflow

1. **Initialization**
   - Load pretrained Flux Kontext model
   - Initialize EEG encoder and adapter
   - Set up LoRA layers on transformer blocks
   - Freeze VAE and text encoders

2. **Data Processing**
   - Load EEG signals and normalize (global mean/std)
   - Encode images to latent space using VAE
   - Prepare latent image IDs for positional encoding

3. **Forward Pass**
   - Encode EEG signals → EEGEncoder → EEG features
   - Convert EEG features → DualEEGAdapter → text embeddings + pooled embeddings
   - Sample noise and timesteps
   - Add noise to latents (flow matching)
   - Concatenate noisy latents with conditional inputs
   - Transformer prediction with LoRA

4. **Training Loss**
   - Flow matching loss: `loss = mean((model_pred - target)²)`
   - Target: `noise - clean_latent`
   - Weighted by timestep-dependent weighting scheme

5. **Optimization**
   - AdamW optimizer
   - Gradient clipping (max_grad_norm=1.0)
   - Trains: LoRA weights, EEG encoder, EEG adapter
   - Frozen: Transformer base weights, VAE, text encoders

### Checkpointing

Checkpoints are saved every 500 steps and include:
- `lora.safetensors`: LoRA weights
- `eeg_encoder_weights.pt`: EEG encoder weights
- `eeg_adapter_weights.pt`: EEG adapter weights

## Model Output

### Saved Checkpoints

```
./style_adapter_eeg_1-80_multi/
├── checkpoint-500/
│   ├── lora.safetensors
│   ├── eeg_encoder_weights.pt
│   └── eeg_adapter_weights.pt
├── checkpoint-1000/
│   ├── lora.safetensors
│   ├── eeg_encoder_weights.pt
│   └── eeg_adapter_weights.pt
└── ...
```

## Inference

To use the trained model for inference:

```python
from diffusers import FluxKontextPipeline
import torch

# Load base model
pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to('cuda')

# Load LoRA weights
pipeline.load_lora_weights(
    './style_adapter_eeg_1-80_multi/checkpoint-12000/',
    weight_name='lora.safetensors'
)

# Load EEG encoder and adapter
eeg_encoder = EEGEncoder()
eeg_encoder.load_state_dict(torch.load('checkpoint-12000/eeg_encoder_weights.pt'))

eeg_adapter = DualEEGAdapter()
eeg_adapter.load_state_dict(torch.load('checkpoint-12000/eeg_adapter_weights.pt'))

# Inference with EEG signals
eeg_signal = load_eeg_data(...)  # Your EEG data
original_image = load_image(...)  # Your input image

# Process EEG
eeg_features = eeg_encoder(eeg_signal)
prompt_embeds, pooled_embeds, text_ids = eeg_adapter_decode(eeg_features, eeg_adapter)

# Generate styled image
image = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_embeds,
    image=original_image,
    guidance_scale=3.5
).images[0]
```

## Advanced Features

### Aspect Ratio Buckets

Support for multiple aspect ratios during training:

```bash
--aspect_ratio_buckets="1024,1024;768,1360;1360,768;880,1168;1168,880"
```

### Text Encoder Fine-tuning

Enable text encoder training (requires more memory):

```bash
--train_text_encoder
--text_encoder_lr=5e-6
```

### Prior Preservation

For better identity preservation:

```bash
--with_prior_preservation
--class_data_dir="path/to/class/images"
--class_prompt="a photo"
--prior_loss_weight=1.0
```

### Experiment Tracking

Enable W&B logging:

```bash
--report_to="wandb"
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `--train_batch_size`
   - Increase `--gradient_accumulation_steps`
   - Use `--gradient_checkpointing`
   - Reduce `--resolution`

2. **NaN Loss**
   - Check EEG normalization statistics
   - Reduce learning rate
   - Enable gradient clipping (already enabled by default)

3. **Slow Training**
   - Use `--cache_latents` to pre-compute VAE latents
   - Reduce `--dataloader_num_workers` if CPU bottleneck
   - Enable `--allow_tf32` on Ampere GPUs

## Custom Components

### EEGEncoder

Located in `Encoder.py`. Processes raw EEG signals into feature representations.

### DualEEGAdapter

Located in `loongx_adapter.py`. Converts EEG features into:
- Sequential embeddings (for cross-attention)
- Global pooled embeddings (for conditioning)

### Custom Layers

Located in `src/layers.py`. Includes:
- `MultiDoubleStreamBlockLoraProcessor`
- `MultiSingleStreamBlockLoraProcessor`

### Pipeline

Located in `src/easycontrol_pipeline.py`:
- `FluxKontextPipeline`: Extended pipeline for EEG conditioning

## Citation

If you use this code, please cite:

```bibtex
@misc{flux-kontext-eeg-style,
  title={EEG-Guided Style Transfer with Flux Kontext},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

Please adhere to the FLUX.1-Kontext-dev licensing terms from Black Forest Labs.

## Acknowledgments

- Built on [Diffusers](https://github.com/huggingface/diffusers) by HuggingFace
- Based on [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) by Black Forest Labs
- Uses LoRA implementation from [PEFT](https://github.com/huggingface/peft)
