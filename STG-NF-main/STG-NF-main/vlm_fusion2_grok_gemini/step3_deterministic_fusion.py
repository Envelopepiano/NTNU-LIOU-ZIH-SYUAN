# =============================================================================
# File: vlm_fusion/step3_deterministic_fusion.py
# Description: 規則式融合 (解決負提升問題)
# =============================================================================

import numpy as np
import os
from sklearn.metrics import roc_auc_score

def run_fusion():
    cache_dir = "vlm_fusion/data_cache"
    print(f"📂 讀取數據中: {cache_dir} ...")
    
    try:
        stg = np.load(os.path.join(cache_dir, "stg_scores.npy")) # Shape (N, ) or (N, 1)
        vlm = np.load(os.path.join(cache_dir, "vlm_scores.npy"))
        gt = np.load(os.path.join(cache_dir, "gt_labels.npy"))
        
        stg = stg.flatten()
        vlm = vlm.flatten()
        gt = gt.flatten()
        
    except FileNotFoundError:
        print("❌ 找不到檔案")
        return

    # 1. STG 分數標準化 (Normalization)
    # STG 原本是 -50 (異常) ~ -0.5 (正常)
    # 我們要把它變成 0 (正常) ~ 1 (異常)
    
    # 針對 Log-Likelihood 的標準化公式：
    # 我們要把 "最小值" (最異常) 映射到 1
    # 把 "最大值" (最正常) 映射到 0
    stg_min = stg.min()
    stg_max = stg.max()
    
    # 公式: (Max - x) / (Max - Min)
    stg_norm = (stg_max - stg) / (stg_max - stg_min)
    
    print(f"📊 STG 正規化後: Min={stg_norm.min():.4f}, Max={stg_norm.max():.4f}")
    
    # 2. 融合策略：Max Pooling (取最大值)
    # 邏輯：STG 覺得異常 OR VLM 覺得異常 = 最終異常
    # 這是為了讓 VLM 能夠 "Rescue" (拯救) STG 漏看的 Case (如腳踏車)
    
    # 你可以調整權重: final = max(stg_norm, vlm * 1.2) 加強 VLM 權重
    final_scores = np.maximum(stg_norm, vlm)

    # 3. 計算 AUC
    baseline_auc = roc_auc_score(gt, stg_norm)
    fused_auc = roc_auc_score(gt, final_scores)
    
    print("\n" + "="*50)
    print(f"✅ 規則融合結果 (Heuristic Fusion)")
    print(f"-----------------------------------")
    print(f"Samples Evaluated : {len(gt)}")
    print(f"VLM Detected      : {np.count_nonzero(vlm)} frames")
    print(f"-----------------------------------")
    print(f"Baseline AUC      : {baseline_auc:.5f}")
    print(f"Fused AUC         : {fused_auc:.5f}")
    
    lift = (fused_auc - baseline_auc) * 100
    if lift > 0:
        print(f"🚀 提升幅度       : +{lift:.4f}% (終於正成長了！)")
    else:
        print(f"📉 提升幅度       : {lift:.4f}%")
    print("="*50)

    # 4. 分析 VLM 到底救了誰？
    # 找出: GT=1 (異常), STG<0.5 (沒抓到), 但 VLM>0.8 (抓到了) 的案例
    rescued_idx = np.where((gt==1) & (stg_norm < 0.5) & (vlm > 0.8))[0]
    print(f"💡 VLM 成功救援 (Rescue) 了 {len(rescued_idx)} 個 Frame")
    if len(rescued_idx) > 0:
        print(f"   例如 Frame ID: {rescued_idx[:5]}")

if __name__ == "__main__":
    run_fusion()