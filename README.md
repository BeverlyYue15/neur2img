# EEG-Guided Style Transfer Training

Training code for EEG-guided style transfer using Flux Kontext with LoRA adapters.

## Overview

Uses EEG (brain) signals to condition style transfer with FLUX.1-Kontext-dev model and LoRA fine-tuning.

**Architecture:**
- **FluxTransformer2DModel**: Base diffusion model
- **EEGEncoder**: Processes EEG signals
- **DualEEGAdapter**: Converts EEG to text embeddings
- **LoRA**: 19 double-stream + 38 single-stream blocks (rank=128)

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, Transformers, Diffusers >= 0.34.0, Accelerate, PEFT

## Dataset Structure

Required files:
- `train_multi.pt`: Packed EEG signals and metadata
- `original.pkl`: Original images
- `transformed.pkl`: Target styled images
- `eeg_mean.npy`, `eeg_std.npy`: Normalization statistics

## Training

### Quick Start

```bash
bash train_style.sh
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--pretrained_model_name_or_path` | `black-forest-labs/FLUX.1-Kontext-dev` | Base model |
| `--output_dir` | `./style_adapter_eeg_1-80_multi` | Output directory |
| `--learning_rate` | 1e-4 | Learning rate |
| `--train_batch_size` | 1 | Batch size per GPU |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation |
| `--max_train_steps` | 12000 | Total training steps |
| `--mixed_precision` | bf16 | Mixed precision |
| `--rank` | 128 | LoRA rank |
| `--resolution` | 512 | Image resolution |
| `--checkpointing_steps` | 500 | Save checkpoint frequency |
| `--use_eeg` | flag | Enable EEG conditioning |

### Accelerate Config

Create `kontext_easycontrol.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 4
gpu_ids: all
```

## Checkpoints

Saved every 500 steps:
```
./style_adapter_eeg_1-80_multi/checkpoint-{step}/
├── lora.safetensors
├── eeg_encoder_weights.pt
└── eeg_adapter_weights.pt
```

## Inference

```python
from diffusers import FluxKontextPipeline
import torch

# Load model
pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to('cuda')

pipeline.load_lora_weights('./checkpoint-12000/', weight_name='lora.safetensors')

# Load EEG models
eeg_encoder.load_state_dict(torch.load('checkpoint-12000/eeg_encoder_weights.pt'))
eeg_adapter.load_state_dict(torch.load('checkpoint-12000/eeg_adapter_weights.pt'))

# Generate
eeg_features = eeg_encoder(eeg_signal)
prompt_embeds, pooled_embeds, text_ids = eeg_adapter_decode(eeg_features, eeg_adapter)
image = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_embeds).images[0]
```

## Troubleshooting

**Out of Memory:** Reduce batch size, increase gradient accumulation, or use `--gradient_checkpointing`

**NaN Loss:** Check EEG normalization or reduce learning rate

**Slow Training:** Use `--cache_latents` or enable `--allow_tf32`

## License

Adhere to FLUX.1-Kontext-dev licensing terms from Black Forest Labs.
