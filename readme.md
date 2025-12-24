# Uni-Neur2Img

<div align="center">

![Uni-Neur2Img](assets/teaser1.png)

**Uni-Neur2Img: Unified Brain Signal-guided Image Generation, Editing, and Stylization**

[![arXiv](https://img.shields.io/badge/arXiv-2512.18635-b31b1b.svg)](https://arxiv.org/abs/2512.18635)

</div>

---

## Introduction

**Uni-Neur2Img** is a unified framework designed for neural signal-driven image generation, editing, and stylization. By leveraging the power of Diffusion Transformers (DiT), our model bridges the gap between human neural activity (EEG) and visual content creation.

**Existing research** in brain-to-image generation often relies on **textual modalities as intermediate representations**, which limits the direct visual expressiveness of neural signals. To address this, **Uni-Neur2Img** introduces a parameter-efficient approach to **directly inject neural signals into the generative process**.


### Key Contributions:
**Unified Framework:** A single architecture that supports multiple tasks: neural-guided image generation, editing, and stylization.

**LoRA-based Neural Injection:** We utilize a parameter-efficient LoRA module to process conditioning signals (EEG, Text, etc.) as pluggable components, allowing for flexible multi-modal conditioning without retraining the base model.

**Causal Attention Mechanism:** Designed to handle the long-sequence modeling requirements inherent in complex neural signal conditioning.

**EEG-Style Dataset**: To facilitate research on visual-modality conditioning, we introduce a new dataset focused on the intersection of EEG signals and artistic styles.

**State-of-the-Art Performance:** Our framework demonstrates superior performance in reconstructing and manipulating images that align with both neural activity and textual descriptions.


📄 **Paper**: [Uni-Neur2Img: Unified Brain Signal-guided Image Generation, Editing, and Stylization](https://arxiv.org/abs/2512.18635)


## Installation

```bash
pip install -r requirements.txt
```

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

