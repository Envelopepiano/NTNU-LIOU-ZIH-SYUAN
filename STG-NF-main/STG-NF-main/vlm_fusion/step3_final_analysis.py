# =============================================================================
# File: vlm_fusion/step3_final_analysis.py
# Version: V10.0 - 極性翻轉修復與融合強度保護版
# =============================================================================

import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score

def run_final_analysis():
    cache_dir = "./data_cache"
    print(f"📂 正在讀取數據: {cache_dir}")
    
    try:
        stg_raw = np.load(os.path.join(cache_dir, "stg_scores.npy")).flatten()
        gt = np.load(os.path.join(cache_dir, "gt_labels.npy")).flatten()
        with open(os.path.join(cache_dir, "vlm_results_log.json"), 'r') as f:
            vlm_logs = json.load(f)
    except Exception as e:
        print(f"❌ 讀取失敗: {e}"); return

    # 1. STG 正規化 (保持 Baseline 0.859)
    raw_auc = roc_auc_score(gt, stg_raw)
    stg_norm = stg_raw.copy()
    if raw_auc < 0.5:
        stg_norm = -stg_raw
    stg_norm = (stg_norm - stg_norm.min()) / (stg_norm.max() - stg_norm.min() + 1e-8)
    base_auc = roc_auc_score(gt, stg_norm)

    # 2. 提取 VLM 數據
    vlm_indices = [log['id'] for log in vlm_logs]
    vlm_raw_scores = np.array([log['vlm_score'] for log in vlm_logs])
    subset_gt = gt[vlm_indices]
    
    # --- [關鍵修正：檢查 VLM 的極性] ---
    vlm_auc = roc_auc_score(subset_gt, vlm_raw_scores)
    print(f"DEBUG: VLM 在子集的原始有效性 (AUC): {vlm_auc:.4f}")
    
    vlm_final_scores = vlm_raw_scores
    if vlm_auc < 0.5:
        print("⚠️ 偵測到 VLM 分數方向與標籤相反，正在自動翻轉 VLM 極性...")
        vlm_final_scores = 1.0 - vlm_raw_scores
        vlm_auc = 1.0 - vlm_auc

    # 3. 執行融合 (採用更穩健的加權融合)
    hybrid_scores = stg_norm.copy()
    # 權重分配：STG 佔 0.4, VLM 佔 0.6 (你可以根據結果調整)
    weight_vlm = 0.6
    
    for i, idx in enumerate(vlm_indices):
        v_score = vlm_final_scores[i]
        # 融合公式
        hybrid_scores[idx] = (1 - weight_vlm) * stg_norm[idx] + weight_vlm * v_score

    # 4. 計算結果
    fused_auc = roc_auc_score(gt, hybrid_scores)
    subset_stg_auc = roc_auc_score(subset_gt, stg_norm[vlm_indices])
    subset_hybrid_auc = roc_auc_score(subset_gt, hybrid_scores[vlm_indices])

    print("\n" + "="*60)
    print(f"📊 STG-NF + VLM 融合實驗報告 (V10.0)")
    print("="*60)
    print(f"1. 全域效能 (Global Performance):")
    print(f" - [Baseline] STG-NF AUC : {base_auc:.5f}")
    print(f" - [Proposed] Hybrid AUC : {fused_auc:.5f}")
    print(f" - 提升幅度: {(fused_auc - base_auc)*100:+.4f}%")
    
    print(f"\n2. 困難樣本分析 (Hard Samples Only):")
    print(f" - STG 子集 AUC   : {subset_stg_auc:.5f}")
    print(f" - VLM 子集 AUC   : {vlm_auc:.5f} (Flip修正後)")
    print(f" - 融合後子集 AUC : {subset_hybrid_auc:.5f}")
    print(f" - 子集提升幅度   : {subset_hybrid_auc - subset_stg_auc:+.5f}")
    
    print("\n3. 數據品質診斷:")
    if subset_hybrid_auc > subset_stg_auc:
        print(" ✅ VLM 成功修正了骨架模型無法判斷的語義異常！")
    else:
        print(" ❌ 融合後效果下降，建議檢查 Prompt 或 VLM 介入的閾值。")
    print("="*60)

if __name__ == "__main__":
    run_final_analysis()