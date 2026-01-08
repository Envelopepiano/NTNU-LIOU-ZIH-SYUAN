# =============================================================================
# File: vlm_fusion/step1_export_real_scores.py
# Description: 
#   這是一個 "Launcher" 腳本。它會呼叫原本的 STG-NF 進行測試，
#   但會透過 "Monkey Patch" 技術攔截評分過程，將分數匯出到 data_cache。
#   完全不需要修改 utils/scoring_utils.py 或 train_eval.py。
# =============================================================================

import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. 設定路徑：讓 Python 找得到上一層的 models, utils 等模組
current_dir = os.path.dirname(os.path.abspath(__file__)) # vlm_fusion/
parent_dir = os.path.dirname(current_dir)                # STG-NF-main/
sys.path.append(parent_dir)

# 2. 匯入原本的模組
import utils.scoring_utils
from train_eval import main as original_main

# 3. 定義我們的 "間諜" 函數 (Hook)
#    這個函數的參數與回傳值必須跟原本的 utils.scoring_utils.score_auc 一模一樣
def spy_score_auc(scores_np, gt):
    print(f"\n[VLM_Fusion] 🕵️  成功攔截到評分數據！")
    print(f"[VLM_Fusion] 分數形狀: {scores_np.shape}, GT 形狀: {gt.shape}")
    
    # 建立快取目錄
    save_dir = os.path.join(current_dir, 'data_cache')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 匯出真實分數
    np.save(os.path.join(save_dir, 'stg_scores.npy'), scores_np)
    np.save(os.path.join(save_dir, 'gt_labels.npy'), gt)
    
    print(f"[VLM_Fusion] ✅ 真實數據已匯出至: {save_dir}")
    print(f"[VLM_Fusion] (stg_scores.npy, gt_labels.npy)")
    print(f"[VLM_Fusion] 下一步：請執行 python vlm_fusion/step2_oracle_test.py 來驗證效果。")
    print("-" * 50)

    # 執行原本的 AUC 計算邏輯 (保持程式正常結束)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc = roc_auc_score(gt, scores_np)
    return auc

# 4. 執行替換 (Monkey Patch)
#    這行指令會把記憶體中原本的 score_auc 換成我們的 spy_score_auc
utils.scoring_utils.score_auc = spy_score_auc

# 5. 模擬命令列參數並啟動程式
if __name__ == "__main__":
    print("[VLM_Fusion] 正在啟動 STG-NF 測試模式 (Hooked)...")
    
    # 模擬你在終端機輸入的參數
    # 如果你的權重檔名不同，請修改下方的 checkpoints/... 路徑
    sys.argv = [
        "train_eval.py", 
        "--mode", "test", 
        "--checkpoint", "checkpoints/ShanghaiTech_85_9.tar",
        "--dataset", "ShanghaiTech"
    ]
    
    # 呼叫原本的主程式
    original_main()