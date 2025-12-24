# Uni-Neur2Img

<div align="center">

![Uni-Neur2Img](assets/teaser.png)

**Uni-Neur2Img: Unified Brain Signal-guided Image Generation, Editing, and Stylization**

[![arXiv](https://img.shields.io/badge/arXiv-2512.18635-b31b1b.svg)](https://arxiv.org/abs/2512.18635)

</div>

---

## Introduction

### Background

Neural style transfer has emerged as a transformative technique in computer vision, enabling the synthesis of images that combine the content of one image with the artistic style of another. However, traditional style transfer methods rely on explicit style reference images, limiting their applicability in scenarios where users wish to express **subjective, cognitive-level aesthetic preferences**.

Recent advances in **Brain-Computer Interfaces (BCI)** have demonstrated that electroencephalogram (EEG) signals can capture rich information about human perception, emotion, and aesthetic preferences. This opens a novel pathway: **using brain signals to directly guide image generation**, bypassing the need for explicit style references.

### Motivation

This project addresses a fundamental question: *Can we decode human aesthetic preferences from EEG signals and use them to condition neural style transfer?*

As illustrated in the teaser figure above, **Uni-Neur2Img** achieves three unified capabilities through brain signal guidance:
- 🎨 **Generation**: Create images from neural signals
- ✏️ **Editing**: Modify existing images based on brain responses  
- 🖼️ **Stylization**: Transfer artistic styles guided by EEG preferences

The key challenges include:
1. **Cross-modal alignment**: Bridging the semantic gap between EEG signals and visual representations
2. **Signal variability**: Handling the high noise and individual differences inherent in EEG data
3. **Integration with generative models**: Efficiently conditioning state-of-the-art diffusion models with EEG features

### Our Approach

We propose an **EEG-Guided Style Transfer** framework that integrates:

| Component | Role |
|-----------|------|
| **EEG Encoder** | Extracts meaningful features from raw EEG signals |
| **Dual EEG Adapter** | Translates EEG features into text embedding space compatible with diffusion models |
| **FLUX.1-Kontext-dev** | State-of-the-art diffusion transformer for high-quality image generation |
| **LoRA Fine-tuning** | Parameter-efficient adaptation (rank=128) preserving base model capabilities |

### Contributions

- **Novel cross-modal conditioning**: First framework to use EEG signals for conditioning Flux-based diffusion models
- **Unified pipeline**: Supports generation, editing, and stylization in one framework
- **Efficient adaptation**: LoRA-based training enabling fine-tuning with limited computational resources
- **End-to-end pipeline**: Complete training and inference code for reproducible research

📄 **Paper**: [Uni-Neur2Img: Unified Brain Signal-guided Image Generation, Editing, and Stylization](https://arxiv.org/abs/2512.18635)

---
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

