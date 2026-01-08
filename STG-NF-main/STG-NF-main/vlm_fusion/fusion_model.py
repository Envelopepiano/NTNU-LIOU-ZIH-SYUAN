# ==========================================
# Date: 2026-01-08
# Author: Gemini (Assisted)
# Description: Lightweight Correction Network (MLP) for fusing STG-NF and VLM scores.
# ==========================================

import torch
import torch.nn as nn

class CorrectionNet(nn.Module):
    def __init__(self):
        super(CorrectionNet, self).__init__()
        
        # Input Dimension = 3
        # 1. STG-NF Score (骨架異常分)
        # 2. VLM Score (語義異常分)
        # 3. Motion/Velocity Score (簡單的速度特徵，可選)
        
        self.model = nn.Sequential(
            nn.Linear(3, 16),      # 第一層：擴展特徵
            nn.BatchNorm1d(16),    # BN 加速收斂
            nn.ReLU(),             # 激活函數
            nn.Dropout(0.1),       # 防止過擬合 (因為數據量可能不大)
            
            nn.Linear(16, 8),      # 第二層
            nn.ReLU(),
            
            nn.Linear(8, 1),       # 輸出層：單一分數
            nn.Sigmoid()           # 壓在 0~1 之間作為機率
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, stg_score, vlm_score, motion_score):
        # 確保輸入維度正確 (Batch, 1)
        x = torch.cat([stg_score, vlm_score, motion_score], dim=1)
        return self.model(x)

if __name__ == "__main__":
    # 測試模型結構
    net = CorrectionNet()
    dummy_input = torch.randn(10, 3) # Batch size 10
    output = net(dummy_input[:, 0:1], dummy_input[:, 1:2], dummy_input[:, 2:3])
    print("Output shape:", output.shape) # 應該是 [10, 1]