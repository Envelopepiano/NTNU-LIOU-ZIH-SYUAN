# ==========================================
# File: vlm_fusion/train_fusion.py
# Description: 最終版 - 適用於真實實驗
# ==========================================

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from importlib import import_module
fusion_module = import_module("fusion_model_g+G")
GatedFusionNet = fusion_module.GatedFusionNet

def load_data():
    cache_dir = "./data_cache"
    print(f"📂 正在讀取數據，路徑: {cache_dir} ...")
    
    try:
        stg = np.load(os.path.join(cache_dir, "stg_scores.npy"))
        vlm = np.load(os.path.join(cache_dir, "vlm_scores.npy"))
        gt = np.load(os.path.join(cache_dir, "gt_labels.npy"))
        
        # 簡單檢查：如果 vlm 全是 0，代表還沒跑 step2
        if np.max(vlm) == 0:
            print("⚠️ 警告：VLM 分數全為 0！這代表融合無效 (等同於只用 STG)。")
            print("請先執行 step2_run_real_vlm_fast.py")

        # 簡單模擬 motion 分數 (保留原邏輯)
        motion = np.random.rand(len(stg), 1).astype(np.float32)
        
        # 轉成 Tensor
        stg_t = torch.tensor(stg, dtype=torch.float32).reshape(-1, 1)
        vlm_t = torch.tensor(vlm, dtype=torch.float32).reshape(-1, 1)
        motion_t = torch.tensor(motion, dtype=torch.float32).reshape(-1, 1)
        gt_t = torch.tensor(gt, dtype=torch.float32).reshape(-1, 1)
        
        return stg_t, vlm_t, motion_t, gt_t
        
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案: {e}")
        exit(1)

def train():
    stg, vlm, motion, gt = load_data()
    
    # 計算 VLM 覆蓋率 (有多少比例是非0的)
    coverage = (np.count_nonzero(vlm.numpy()) / len(vlm)) * 100
    print(f"📊 VLM 覆蓋率: {coverage:.2f}% (這是真正送給 GPT 檢查的比例)")

    # 2. 初始化模型 - 使用新的 GatedFusionNet
    model = GatedFusionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 加 L2 reg 防止過擬合
    criterion = nn.BCELoss()
    
    print(f"🚀 開始訓練 Gated Attention Fusion Network (Data size: {len(gt)})...")
    
    # 3. 訓練
    for epoch in range(150):  # 訓練稍微久一點
        model.train()
        optimizer.zero_grad()
        
        # Forward 會有兩個輸出: predictions 和 attention_weights
        preds, attention_weights = model(stg, vlm, motion)
        
        # Loss 1: 分類誤差
        cls_loss = criterion(preds, gt)
        
        # (Optional) Loss 2: Sparsity Loss
        # 我們希望 Attention 盡量明確，不要總是 0.5
        # attn_loss = torch.mean(torch.abs(attention_weights - 0.5)) * 0.1
        
        total_loss = cls_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # 觀察 VLM 的介入程度
            avg_attn = attention_weights.mean().item()
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}, Avg VLM Attention: {avg_attn:.3f}")
            
    # 4. 評估
    model.eval()
    with torch.no_grad():
        final_preds, final_attn = model(stg, vlm, motion)
        
        y_true = gt.numpy()
        y_scores = final_preds.numpy()
        stg_baseline = stg.numpy()
        
        auc = roc_auc_score(y_true, y_scores)
        stg_auc = roc_auc_score(y_true, stg_baseline)
        
        # 分析 Attention 分佈
        avg_final_attn = final_attn.mean().item()
        
        print("\n" + "="*50)
        print(f"✅ 真實實驗結果 (Gated Attention Fusion)")
        print(f"Baseline (STG-NF Original): {stg_auc:.4f}")
        print(f"Fused (Hybrid Framework)  : {auc:.4f}")
        print(f"提升幅度: {(auc - stg_auc)*100:.2f}%")
        print(f"平均 VLM Attention Weight : {avg_final_attn:.3f}")
        print("="*50)

if __name__ == "__main__":
    train()