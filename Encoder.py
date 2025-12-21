# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from s4torch import S4Model

# def _check(t, name):
#         if torch.isnan(t).any() or torch.isinf(t).any():
#             raise RuntimeError(f"[NaN/Inf] {name}  -->  min={t.min().item():.3f}  max={t.max().item():.3f}")
#         return t
# # ---- 时序金字塔池化函数 ----
# def spatial_pyramid_pooling(x, output_size, adaptive=False):
#     """
#     Convert variable-length EEG sequence to fixed length (e.g., 1000 -> 4096)
#     x: [B, C, L]
#     return: [B, C, output_size]
#     """
#     B, C, L = x.shape
#     if L == output_size:
#         return x
#     if adaptive:
#         return F.adaptive_avg_pool1d(x, output_size)
#     else:
#         if L < output_size:
#             pad = torch.zeros(B, C, output_size - L, device=x.device, dtype=x.dtype)
#             return torch.cat([x, pad], dim=2)
#         else:
#             return x[:, :, :output_size]


# # ✅ ---- 修正版 FeaturePyramidPooling ----
# class FeaturePyramidPooling(nn.Module):
#     """
#     Multi-scale pooling for EEG feature extraction.
#     支持 in/out 通道映射，确保与主通道一致。
#     """
#     def __init__(self, output_sizes=[128, 256, 512, 1024, 2048],
#                  in_channels=16, out_channels=16):
#         super().__init__()
#         self.output_sizes = output_sizes
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # ✅ 通道映射层：保证输出与主干通道一致
#         self.channel_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#         self.pools = nn.ModuleList([
#             nn.AdaptiveAvgPool1d(size) for size in output_sizes
#         ])

#     def forward(self, x):
#         """
#         x: [B, in_channels, L]
#         return: [B, out_channels, sum(output_sizes)]
#         """
#         x = self.channel_proj(x)  # [B, out_channels, L]
#         pooled = [pool(x) for pool in self.pools]
#         return torch.cat(pooled, dim=-1)  # 拼接为 (B, out_channels, 3968)


# # ✅ ---- EEGEncoder ----

# class EEGEncoder(nn.Module):
#     def __init__(
#         self,
#         output_shape=(16, 64, 64), # (C_out, H, W)
#         device: str = "cuda",
#         dtype: torch.dtype = torch.bfloat16,
#         s4_d_model: int = 256, # S4的隐藏维度
#     ):
#         super().__init__()
#         self.C_out, self.H, self.W = output_shape
#         self.s4_d_model = s4_d_model
#         self.target_length = self.H * self.W # 64*64 = 4096
#         self.device = device
#         self.dtype = dtype

#         # 标志位，用于延迟初始化
#         self.initialized = False

#     def _create_modules(self, d_input, seq_len):
#         """根据实际输入动态创建所有模块"""
#         # 1. S4 模型: 提取深度时序特征
#         self.s4 = S4Model(
#             d_input=d_input,
#             d_model=self.s4_d_model,
#             d_output=self.s4_d_model,
#             n_blocks=2,
#             n=self.s4_d_model,
#             l_max=seq_len
#         ).to(self.device)

#         # 2. 线性层: 将 S4 输出映射到目标长度
#         # 输入: [B, s4_d_model, seq_len]
#         # 输出: [B, s4_d_model, target_length(4096)]
#         self.length_adapter = nn.Linear(seq_len, self.target_length).to(self.device).to(self.dtype)

#         # 3. 卷积块: 将 1D 特征重塑并精炼为 2D 特征图
#         # 输入: [B, s4_d_model, 64, 64]
#         # 输出: [B, 16, 64, 64]
#         self.conv_block = nn.Sequential(
#             # 第一个卷积层，调整通道数并加入非线性
#             nn.Conv2d(self.s4_d_model, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # 第二个卷积层，进一步精炼
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             # 最终输出层
#             nn.Conv2d(32, self.C_out, kernel_size=1) # 1x1 conv to get to 16 channels
#         ).to(self.device).to(self.dtype)

#         self.initialized = True

#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape [B, C_in, L_in] e.g., [B, 59, 1000]
#         Returns:
#             Output tensor of shape [B, 16, 64, 64]
#         """
#         B, C_in, L_in = x.shape
        
#         # --- 动态初始化 ---
#         if not self.initialized:
#             self._create_modules(d_input=C_in, seq_len=L_in)

#         # --- 1. 通过 S4 提取时序特征 ---
#         # S4 expects [B, L, C]
#         z = x.permute(0, 2, 1).contiguous() # [B, L_in, C_in]
#         z = z.to(torch.float32) # S4 needs float32
#         z = self.s4(z) # [B, L_in, s4_d_model]
#         z = z.to(x.dtype)
#         z = z.permute(0, 2, 1).contiguous() # [B, s4_d_model, L_in]

#         # --- 2. 调整序列长度到 4096 (64*64) ---
#         # Use a linear layer to map from L_in to 4096
#         z = self.length_adapter(z) # [B, s4_d_model, 4096]

#         # --- 3. 重塑为 2D 特征图 [B, s4_d_model, 64, 64] ---
#         z = z.view(B, self.s4_d_model, self.H, self.W) # [B, s4_d_model, 64, 64]

#         # --- 4. 通过卷积块精炼特征并调整通道数 ---
#         z.to(x.dtype)
#         out = self.conv_block(z) # [B, 16, 64, 64]
#         # z.to(x.dtype)
#         return out
# # class EEGEncoder(nn.Module):
# #     def __init__(self, device="cuda", dtype=torch.bfloat16):
# #         super().__init__()
# #         self.device = device
# #         self.dtype = dtype
# #         self.eeg_fixed_length = 4096

# #         # 输入适配：59 -> 16
# #         self.channel_adapter = nn.Conv1d(59, 16, kernel_size=1).to(device).to(dtype)

# #         # S4 模块
# #         d_input1 = d_input2 = 16
# #         self.s41 = S4Model(d_input1, d_model=64, d_output=64, n_blocks=2,
# #                            n=64, l_max=self.eeg_fixed_length).to(device)
# #         self.s42 = S4Model(d_input2, d_model=16, d_output=16, n_blocks=2,
# #                            n=16, l_max=self.eeg_fixed_length).to(device)

# #         self.pool1 = nn.AdaptiveAvgPool1d(4).to(device).to(dtype)
# #         self.pool2 = nn.AdaptiveAvgPool1d(64).to(device).to(dtype)

# #         # ✅ 通道映射层：把 z1 的 4 通道 -> 16 通道
# #         self.z1_channel_proj = nn.Conv1d(4, 16, kernel_size=1).to(device).to(dtype)

# #         # FPP 模块
# #         self.fpp = FeaturePyramidPooling(
# #             output_sizes=[128, 256, 512, 1024, 2048],
# #             in_channels=16, out_channels=16
# #         ).to(device).to(dtype)

# #         # 投影层
# #         self.projection = nn.Sequential(
# #             nn.Conv1d(16, 16, kernel_size=1),
# #             nn.GroupNorm(16, 16),
# #             nn.GELU()
# #         ).to(device).to(dtype)
        
    

# #     def forward(self, x):
# #         B = x.size(0)
# #         x = x.to(self.device).to(self.dtype)
# #         _check(x, "input")

# #         # 1. 通道适配
# #         x = self.channel_adapter(x)                # (B, 16, 1000)
# #         _check(x, "after channel_adapter")

# #         # 2. 时长统一
# #         x = spatial_pyramid_pooling(x, output_size=self.eeg_fixed_length, adaptive=True)
# #         _check(x, "after SPP")
# #         print(f"[S4-in] min={x.permute(0,2,1).min().item():.4f}  max={x.permute(0,2,1).max().item():.4f}")
# #         # 3. 两路 S4
# #         z1 = self.s41(x.permute(0, 2, 1)).permute(0, 2, 1)
# #         _check(z1, "after s41")
# #         z1 = self.pool1(z1).permute(0, 2, 1)
# #         _check(z1, "after pool1")
# #         z1 = self.z1_channel_proj(z1)
# #         _check(z1, "after z1_channel_proj")

# #         z2 = self.s42(x.permute(0, 2, 1)).permute(0, 2, 1)
# #         _check(z2, "after s42")
# #         z2 = self.pool2(z2)
# #         _check(z2, "after pool2")

# #         # 4. FPP
# #         x_fpp = self.fpp(x)
# #         _check(x_fpp, "after fpp")

# #         # 5. 拼接 + 投影
# #         x_combined = torch.cat([z1, x_fpp, z2], dim=-1)
# #         _check(x_combined, "after cat")
# #         x_proj = self.projection(x_combined)
# #         _check(x_proj, "after projection")

# #         # 6. reshape
# #         x_out = x_proj.view(B, 16, 64, 64)
# #         _check(x_out, "final output")
# #         return x_out
# #     def forward(self, x):
# #         """
# #         输入: (B, 59, 1000)
# #         输出: (B, 16, 64, 64)
# #         """
# #         B = x.size(0)
# #         x = x.to(self.device).to(self.dtype)
        
# #         # Step 1: 通道适配
# #         x = self.channel_adapter(x)  # (B, 16, 1000)

# #         # Step 2: 时长统一
# #         x = spatial_pyramid_pooling(x, output_size=self.eeg_fixed_length, adaptive=True)  # (B, 16, 4096)

# #         # Step 3: 两路 S4
# #         z1 = self.s41(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 64, 4096)
# #         z1 = self.pool1(z1).permute(0, 2, 1)                # (B, 4, 64)
# #         z1 = self.z1_channel_proj(z1)                       # ✅ (B, 16, 64)

# #         z2 = self.s42(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 16, 4096)
# #         z2 = self.pool2(z2)                                 # (B, 16, 64)

# #         # Step 4: FPP
# #         x_fpp = self.fpp(x)                                 # (B, 16, 3968)

# #         # Step 5: 拼接 + 投影
# #         x_combined = torch.cat([z1, x_fpp, z2], dim=-1)     # ✅ (B, 16, 4096)
# #         x_proj = self.projection(x_combined)                # (B, 16, 4096)

# #         # Step 6: reshape 成 2D latent
# #         x_out = x_proj.view(B, 16, 64, 64)                  # ✅ (B, 16, 64, 64)
# #         return x_out
from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F
from s4torch import S4Model

# --------------------------------------------------
# 1. 工具：时序金字塔池化 + 保守初始化
# --------------------------------------------------
def spatial_pyramid_pooling(x, output_size, adaptive=False):
    B, C, L = x.shape
    if L == output_size:
        return x
    if adaptive:
        return F.adaptive_avg_pool1d(x, output_size)
    else:
        if L < output_size:
            pad = torch.zeros(B, C, output_size - L, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=2)
        else:
            return x[:, :, :output_size]

def init_small(m):
    """保守初始化：线性层 Xavier 增益 1e-2，卷积层同理"""
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=1e-2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# --------------------------------------------------
# 2. FeaturePyramidPooling
# --------------------------------------------------
class FeaturePyramidPooling(nn.Module):
    def __init__(self, output_sizes=(128, 256, 512, 1024, 2048),
                 in_channels=16, out_channels=16):
        super().__init__()
        self.channel_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(s) for s in output_sizes])
        self.apply(init_small)

    def forward(self, x):
        x = self.channel_proj(x)
        pooled = [p(x) for p in self.pools]
        return torch.cat(pooled, dim=-1)          # [B, out_channels, sum(output_sizes)]

# --------------------------------------------------
# 3. EEG 编码器
# --------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(self,
                 eeg_fixed_length: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype=torch.float32):
        super().__init__()
        self.eeg_fixed_length = eeg_fixed_length
        self.dtype = dtype
        self.device = device

        # --- 分支 1：59 通道 → 64 维 ---
        self.s41 = S4Model(d_input=59, d_model=64, d_output=64,
                           n_blocks=2, n=64, l_max=eeg_fixed_length).to(device)

        # --- 分支 2：仅取 4 通道 → 4 维 ---
        self.s42 = S4Model(d_input=4, d_model=4, d_output=4,
                           n_blocks=2, n=4, l_max=eeg_fixed_length).to(device)

        # --- 池化 & 特征金字塔 ---
        self.pool1 = nn.AdaptiveAvgPool1d(4).to(dtype)
        self.pool2 = nn.AdaptiveAvgPool1d(64).to(dtype)
        self.fpp = FeaturePyramidPooling(output_sizes=(128, 256, 512, 1024, 2048),
                                         in_channels=59, out_channels=16).to(device, dtype)

        # --- 投影头 ---
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),                # 64000
            nn.Linear(64000, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Unflatten(1, (512, 8)),
            nn.Linear(8, 128)
        ).to(device, dtype)
        self.head = nn.Sequential(
            nn.Linear(128, 64 * 64),            # 128 → 4096
            nn.Unflatten(2, (64, 64)),          # [B, 512, 64, 64]
            nn.Conv2d(512, 16, kernel_size=1, bias=True)
        ).to(device, dtype)

        self.apply(init_small)

    # ========== 前向：逐层 NaN 检查 ==========
    def forward(self, x: torch.Tensor):
        x = x.to(dtype=self.dtype)
        x = spatial_pyramid_pooling(x, self.eeg_fixed_length)
        _check(x, "after SPP")

        # ---- 分支 1 ----
        z1 = x.permute(0, 2, 1).contiguous()
        z1 = self.s41(z1).to(self.dtype)
        _check(z1, "after s41")
        z1 = z1.permute(0, 2, 1).contiguous()
        z1 = self.pool1(z1)
        z1 = z1.permute(0, 2, 1).contiguous()          # [B, 4, 64]
        _check(z1, "after pool1")

        # ---- 分支 2 ----
        z2 = x[:, :4, :].permute(0, 2, 1).contiguous()
        z2 = self.s42(z2).to(self.dtype)
        _check(z2, "after s42")
        z2 = z2.permute(0, 2, 1).contiguous()
        z2 = self.pool2(z2)                            # [B, 64, 4]
        _check(z2, "after pool2")

        # ---- 特征金字塔 ----
        x_fpp = self.fpp(x)
        _check(x_fpp, "after FPP")

        # ---- 拼接 ----
        flat = torch.cat([z1.flatten(1), z2.flatten(1), x_fpp.flatten(1)], dim=1)
        _check(flat, "after cat")
        # print(flat.shape)
        # ---- 投影 ----
        out = self.projection(flat)
        _check(out, "final output")
        out_final=self.head(out)
        return out_final

# --------------------------------------------------
# 4. 辅助：遇到 NaN 直接报错
# --------------------------------------------------
def _check(t: torch.Tensor, tag: str):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaN/Inf] detected {tag}! shape={t.shape}")