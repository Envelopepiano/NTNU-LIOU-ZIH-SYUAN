# fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusionNet(nn.Module):
    def __init__(self, input_dim=3):
        super(GatedFusionNet, self).__init__()
        
        # 特徵提取器 (Feature Extractor)
        # 把 STG 和 VLM 分數投射到高維空間
        self.stg_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        self.vlm_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # 門控網路 (Gating Network) - 注意力機制核心
        # 輸入: 兩個模態的特徵 + Motion (如有)
        # 輸出: 一個 0~1 的權重 alpha
        # 如果 alpha 接近 1，表示相信 VLM；接近 0 表示相信 STG
        self.attention_gate = nn.Sequential(
            nn.Linear(32 + 1, 16), # 32(feat) + 1(motion)
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
        # 最終預測層
        self.classifier = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, stg_score, vlm_score, motion_score):
        # 1. 特徵編碼
        stg_feat = self.stg_encoder(stg_score)  # Shape: [B, 16]
        vlm_feat = self.vlm_encoder(vlm_score)  # Shape: [B, 16]
        
        # 2. 計算注意力權重 (Gating Weight)
        # 我們把特徵串接起來，讓網路判斷當前的狀況
        concat_feat_for_gate = torch.cat([stg_feat, vlm_feat, motion_score], dim=1)
        alpha = self.attention_gate(concat_feat_for_gate) # Alpha: 信任 VLM 的程度
        
        # 3. 特徵融合 (Weighted Fusion)
        # 這是本文的創新點：Semantic-Guided Refinement
        # 融合特徵 = (1-alpha) * STG特徵 + alpha * VLM特徵
        fused_feat = (1 - alpha) * stg_feat + alpha * vlm_feat
        
        # 也可以選擇把原始分數也加權 (Residual Connection)
        # fused_score_raw = (1 - alpha) * stg_score + alpha * vlm_score
        
        # 4. 最終分類
        # 我們把融合後的特徵再過一層分類器，確保非線性能力
        # 同時把原始的融合特徵也串進去 (Skip Connection) 來保留原始特徵
        final_input = torch.cat([stg_feat, fused_feat], dim=1) # 16+16=32
        prediction = self.classifier(final_input)
        
        return prediction, alpha # 回傳 alpha 是為了分析 VLM 何時介入

if __name__ == "__main__":
    net = GatedFusionNet()
    dummy_stg = torch.randn(10, 1)
    dummy_vlm = torch.randn(10, 1)
    dummy_mot = torch.randn(10, 1)
    out, attn = net(dummy_stg, dummy_vlm, dummy_mot)
    print(f"Output: {out.shape}, Attention Weights: {attn.mean().item():.3f}")