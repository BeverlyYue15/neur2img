import torch
import torch.nn as nn

class DualEEGAdapter(nn.Module):
    """
    EEG [batch, 16, 64, 64] -> 文本嵌入
    支持两种输出格式：
      - 全局嵌入: [batch, 768]
      - 序列嵌入: [batch, 512, 4096]
    """
    
    def __init__(self, eeg_channels=16):
        super().__init__()
        
        # 全局输出路径：空间池化 + 映射
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # [B, 16, 1, 1]
            nn.Flatten(),                      # [B, 16]
            nn.Linear(eeg_channels, 768),      # [B, 768]
            nn.LayerNorm(768)
        )
        
        # 序列输出路径：调整通道数 + 展平空间维度
        self.seq_path = nn.Sequential(
            nn.Conv2d(eeg_channels, 512, 1),   # [B, 512, 64, 64]
            nn.ReLU(),
            nn.Flatten(2),                     # [B, 512, 4096]
            nn.LayerNorm(4096)
        )
        
    def forward(self, eeg, output_type="global"):
        """
        Args:
            eeg: [batch, 16, 64, 64]
            output_type: "global" 或 "seq"
        """
        if output_type == "global":
            return self.global_path(eeg)       # [B, 768]
        elif output_type == "seq":
            return self.seq_path(eeg)          # [B, 512, 4096]
        else:
            raise ValueError("output_type must be 'global' or 'seq'")

# # ==================== 使用示例 ====================
# if __name__ == "__main__":
#     # 模拟输入
#     eeg = torch.randn(1, 16, 64, 64)
    
#     # 初始化适配器
#     adapter = DualEEGAdapter()
    
#     # 映射到全局文本嵌入 (cls token风格)
#     global_emb = adapter(eeg, output_type="global")
#     print(f"全局嵌入形状: {global_emb.shape}")  # torch.Size([1, 768])
    
#     # 映射到序列文本嵌入 (token序列风格)
#     seq_emb = adapter(eeg, output_type="seq")
#     print(f"序列嵌入形状: {seq_emb.shape}")    # torch.Size([1, 512, 4096])
    
#     # 验证输出分布是否稳定
#     print(f"全局嵌入均值: {global_emb.mean():.4f}, 方差: {global_emb.var():.4f}")
#     print(f"序列嵌入均值: {seq_emb.mean():.4f}, 方差: {seq_emb.var():.4f}")


# import torch
# import torch.nn as nn

# class Conv1dEEGAdapter(nn.Module):
#     """
#     使用 Conv1d + LayerNorm 的EEG适配器
#     输入: [batch, 16, 64, 64]
#     输出: 全局 [batch, 768] 或 序列 [batch, 512, 4096]
#     """
    
#     def __init__(self, eeg_channels=16):
#         super().__init__()
        
#         # 展平空间维度: [B, 16, 64, 64] -> [B, 16, 4096]
#         self.flatten = nn.Flatten(2)
        
#         # 全局路径：压缩空间维度 -> 映射到768
#         self.global_path = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),                    # [B, 16, 1]
#             nn.Flatten(),                               # [B, 16]
#             nn.LayerNorm(16),                           # 预归一化
#             nn.Linear(16, 768),                         # [B, 768]
#             nn.LayerNorm(768)
#         )
        
#         # 序列路径：直接通道转换
#         self.seq_path = nn.Sequential(
#             nn.LayerNorm(4096),                         # 先对特征维度归一化
#             nn.Conv1d(eeg_channels, 512, kernel_size=1), # [B, 512, 4096]
#             nn.LayerNorm(4096)                          # 再次归一化
#         )
        
#     def forward(self, eeg, output_type="global"):
#         """
#         Args:
#             eeg: [batch, 16, 64, 64]
#             output_type: "global" 或 "seq"
#         """
#         # 统一展平: [B, 16, 4096]
#         x = self.flatten(eeg)
        
#         if output_type == "global":
#             return self.global_path(x)   # [B, 768]
        
#         elif output_type == "seq":
#             return self.seq_path(x)      # [B, 512, 4096]

# ==================== 使用示例 ====================
# if __name__ == "__main__":
#     # 模拟数据
#     eeg = torch.randn(1, 16, 64, 64)
    
#     # 初始化
#     adapter = Conv1dEEGAdapter()
    
#     # 测试两种输出
#     global_emb = adapter(eeg, "global")
#     seq_emb = adapter(eeg, "seq")
    
#     print(f"EEG 输入: {eeg.shape}")               # [1, 16, 64, 64]
#     print(f"全局嵌入: {global_emb.shape}")        # [1, 768]
#     print(f"序列嵌入: {seq_emb.shape}")           # [1, 512, 4096]
    
#     # 验证数值稳定性
#     print(f"\n全局 - 均值: {global_emb.mean():.4f}, 方差: {global_emb.var():.4f}")
#     print(f"序列 - 均值: {seq_emb.mean():.4f}, 方差: {seq_emb.var():.4f}")