# vlm_fusion/train_fusion.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from fusion_model import CorrectionNet

def load_data():
    print("正在讀取 data_cache 中的數據...")
    try:
        stg = np.load("data_cache/stg_scores.npy")
        vlm = np.load("data_cache/vlm_scores.npy")
        gt = np.load("data_cache/gt_labels.npy")
        
        # 模擬一個簡單的 motion 分數 (真實情況你要另外算)
        motion = np.random.rand(len(stg), 1).astype(np.float32)
        
        # 轉成 Tensor
        stg_t = torch.tensor(stg, dtype=torch.float32)
        vlm_t = torch.tensor(vlm, dtype=torch.float32)
        motion_t = torch.tensor(motion, dtype=torch.float32)
        gt_t = torch.tensor(gt, dtype=torch.float32)
        
        return stg_t, vlm_t, motion_t, gt_t
        
    except FileNotFoundError:
        print("錯誤：找不到數據檔案，請確認步驟一和步驟二已執行。")
        exit()

def train():
    # 1. 準備數據
    stg, vlm, motion, gt = load_data()
    
    # 簡單分割 80% 訓練, 20% 測試
    split = int(0.8 * len(gt))
    train_data = (stg[:split], vlm[:split], motion[:split], gt[:split])
    test_data = (stg[split:], vlm[split:], motion[split:], gt[split:])
    
    # 2. 初始化模型
    model = CorrectionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print(f"開始訓練融合網路 (Train size: {len(train_data[0])})...")
    
    # 3. 訓練迴圈
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        preds = model(train_data[0], train_data[1], train_data[2])
        loss = criterion(preds, train_data[3])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 4. 評估
    model.eval()
    with torch.no_grad():
        final_preds = model(test_data[0], test_data[1], test_data[2])
        
        y_true = test_data[3].numpy()
        y_scores = final_preds.numpy()
        stg_baseline = test_data[0].numpy()
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            stg_auc = roc_auc_score(y_true, stg_baseline)
            
            print("\n" + "="*40)
            print(f"✅ 訓練完成！")
            print(f"Baseline (STG-NF) AUC: {stg_auc:.4f}")
            print(f"Fused (VLM-Corrected) AUC: {auc:.4f}")
            print(f"提升幅度: {(auc - stg_auc)*100:.2f}%")
            print("="*40)
        except ValueError:
            print("警告：測試集樣本中只有單一類別，無法計算 AUC。")

if __name__ == "__main__":
    train()