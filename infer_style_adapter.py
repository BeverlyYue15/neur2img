import os
import torch
from PIL import Image
from src_adapter.easycontrol_pipeline import FluxKontextPipeline
from src_adapter.transformer_flux import FluxTransformer2DModel
from src_adapter.lora_helper import set_multi_lora
import torch.nn.functional as F
from train_lora_flux_kontext_easycontrol import EEGImageToImageDataset
# 可选：你自己的 EEGEncoder
from Encoder import EEGEncoder  # 与训练一致
from pathlib import Path
from loongx_adapter import  DualEEGAdapter


def eeg_adapter_decode(
    eeg,
    adapter
):
    
    prompt_embeds = adapter(eeg, output_type="seq").to(dtype=torch.bfloat16,device="cuda:0") 
    
    pooled_prompt_embeds = adapter(eeg, output_type="global").to(dtype=torch.bfloat16,device="cuda:0") 
    
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=torch.bfloat16,device="cuda:0") 
    
    return prompt_embeds, pooled_prompt_embeds, text_ids



def run_infer(
    device="cuda",
    base="black-forest-labs/FLUX.1-Kontext-dev",
    lora_dir="style_adapter_eeg_1-80/checkpoint-4500/",
    ori_img="path/to/ori.png",
    eeg_pt="path/to/eeg_tensor.pt",
    prompt="Using the provided EEG signals to perform style transfer.",
    height=1024, width=1024, cond_size=512,
    steps=25, guidance=3.5, seed=1234,
):
    dtype = torch.bfloat16
    pipe = FluxKontextPipeline.from_pretrained(base, torch_dtype=dtype)
    transformer = FluxTransformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.to(device)
    pipe.transformer.eval(); pipe.vae.eval()

    # LoRA
    set_multi_lora(pipe.transformer, [os.path.join(lora_dir, "lora.safetensors")], lora_weights=[[1]], cond_size=cond_size)

    # 准备原图
    

    # 计算 cond 网格尺寸（与你 pipeline 的 prepare_latents 对齐）
    vae_sf = pipe.vae_scale_factor
    # h_cond = 2 * (int(cond_size) // (vae_sf * 2))
    # w_cond = 2 * (int(cond_size) // (vae_sf * 2))

    # EEG → latent
    eeg_encoder = EEGEncoder(device=device, dtype=torch.float32)
    eeg_adapter = DualEEGAdapter().to(dtype=torch.bfloat16,device="cuda")
    eeg_ckpt = Path(lora_dir) / "eeg_encoder_weights.pt"
    if not eeg_ckpt.exists():
        raise FileNotFoundError(f"找不到 EEG 权重：{eeg_ckpt}")
    state_dict = torch.load(eeg_ckpt, map_location=device)
    # 去掉可能的 module. 前缀
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    eeg_encoder.load_state_dict(clean_sd, strict=True)
    eeg_encoder.eval()
    eeg_adapter_ckpt = Path(lora_dir) / "eeg_adapter_weights.pt"
    if not eeg_adapter_ckpt.exists():
        raise FileNotFoundError(f"找不到 EEG 权重：{eeg_adapter_ckpt}")
    state_dict = torch.load(eeg_adapter_ckpt, map_location=device)
    # 去掉可能的 module. 前缀
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    eeg_adapter.load_state_dict(clean_sd, strict=True)
    eeg_adapter.eval()
    
    ds = EEGImageToImageDataset("1108_04valstyle/val1-80.pt")
    # id=2
    for id in range(149,len(ds)):
        eeg = ds[id]["eeg_values"]          # 从 dataset 拿一条 EEG
        pic_name=ds[id]["label"]
        print(pic_name)
        # prompt=ds[id]["prompts"]
        prompt=""
        ori = Image.open("original/"+pic_name+".png").convert("RGB")
        eeg_latent=eeg_encoder(eeg.unsqueeze(0).to("cuda:0")).to(dtype=torch.bfloat16,device="cuda:0") 
        prompt_embeds, pooled_prompt_embeds, text_ids=eeg_adapter_decode(eeg_latent,eeg_adapter)
        g = torch.Generator(device=device).manual_seed(seed)
        for i in range(1):
            out = pipe(
                image=ori,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                max_sequence_length=512,
                cond_size=cond_size,
                subject_images=[eeg_latent],          # 不用 subject
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                # eeg_latents=eeg_lat,        # ← 新增的 EEG 条件
                # subject_is_latents=False,   # 对 subject_images 生效；这里留默认
                # generator=g,
            ).images[0]
            out.save(f"results/style_eeg/{pic_name}.png")
            print(f"saved results_eeg_{i}.png")

if __name__ == "__main__":
    run_infer()
