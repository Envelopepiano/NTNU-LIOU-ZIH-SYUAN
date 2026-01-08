# =============================================================================
# File: vlm_fusion/step2_oracle_test.py
# Description: 
#   生成 "完美的" VLM 分數。這是為了驗證你的架構潛力 (Upper Bound Analysis)。
#   它會偷看 Ground Truth，並產生跟 GT 高度相關的分數 (模擬 VLM 100% 看懂異常)。
# =============================================================================

import numpy as np
import os

def generate_oracle_vlm():
    print("正在生成 '完美 VLM' 分數 (僅用於驗證架構潛力)...")
    
    # 1. 讀取剛剛匯出的真實 GT (Ground Truth)
    try:
        gt = np.load("vlm_fusion/data_cache/gt_labels.npy")
        stg = np.load("vlm_fusion/data_cache/stg_scores.npy")
    except FileNotFoundError:
        # 如果路徑不對，試試看相對路徑
        try:
            gt = np.load("data_cache/gt_labels.npy")
            stg = np.load("data_cache/stg_scores.npy")
        except FileNotFoundError:
            print("錯誤：找不到真實數據！請確認 vlm_fusion/data_cache/ 裡面有 .npy 檔")
            return

    print(f"載入數據: {len(gt)} 筆樣本")

    # 2. 模擬一個完美的 VLM
    # 邏輯：如果 GT 是異常 (1)，VLM 就給高分 (0.9~1.0)
    #       如果 GT 是正常 (0)，VLM 就給低分 (0.0~0.1)
    
    # 複製 GT
    vlm_scores = gt.copy()
    
    # 加一點點隨機雜訊 (Noise)，讓它看起來不像作弊那麼假，而是像一個"很強的模型"
    noise = np.random.normal(0, 0.05, vlm_scores.shape) # 5% 的雜訊
    vlm_scores = vlm_scores + noise
    
    # 限制分數在 0~1 之間
    vlm_scores = np.clip(vlm_scores, 0, 1)

    # 3. 儲存結果
    # 存到同一個 cache 資料夾，這樣 train_fusion.py 就可以直接讀
    save_path = "vlm_fusion/data_cache/vlm_scores.npy"
    if not os.path.exists("vlm_fusion/data_cache"):
        save_path = "data_cache/vlm_scores.npy"
        
    np.save(save_path, vlm_scores)
    print(f"✅ 完美 VLM 分數已生成至: {save_path}")
    print("現在去跑 python vlm_fusion/train_fusion.py，AUC 應該會飆升！")

if __name__ == "__main__":
    generate_oracle_vlm()