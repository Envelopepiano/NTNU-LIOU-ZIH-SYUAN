# =============================================================================
# File: vlm_fusion/step1_5_get_file_mapping.py
# Description: 
#   修正版 (Method 2)：基於 scoring_utils 的邏輯來建立對照表。
#   直接遍歷 GT 目錄，確保生成的 Mapping 與 stg_scores.npy (Frame-level) 完全對齊。
# =============================================================================

import os
import numpy as np
from tqdm import tqdm

def extract_mapping_from_gt():
    # 1. 設定路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 回到 STG-NF-main/
    # 這是存放 Ground Truth 遮罩的地方，每個檔案代表一個測試影片
    gt_dir = os.path.join(base_dir, 'data', 'ShanghaiTech', 'gt', 'test_frame_mask')
    
    print(f"🚀 正在依照 GT 目錄建立對照表: {gt_dir}")
    
    if not os.path.exists(gt_dir):
        print(f"❌ 錯誤：找不到目錄 {gt_dir}")
        print("請確認你的 data 資料夾結構是否正確。")
        return

    # 2. 獲取並排序影片列表 (邏輯必須跟 scoring_utils.py 一模一樣)
    # 參考 scoring_utils.py line 45: sorted(fn for fn in clip_list if fn.endswith('.npy'))
    clip_list = sorted([fn for fn in os.listdir(gt_dir) if fn.endswith('.npy')])
    
    print(f"📊 找到 {len(clip_list)} 個測試影片片段")
    
    mapping_list = []
    
    # 3. 遍歷影片，生成每一幀的 ID
    for clip_name in tqdm(clip_list):
        # clip_name 範例: "01_0014.npy"
        # 解析 Video ID (參考 scoring_utils.py line 85)
        # 這裡我們直接存檔名即可，方便 step2 解析
        # 格式: 01_014 (Scene 01, Clip 014)
        
        # 讀取 GT 檔只是為了知道 "這影片有幾幀"
        gt_path = os.path.join(gt_dir, clip_name)
        gt_data = np.load(gt_path)
        num_frames = len(gt_data)
        
        # 提取可讀的 ID，例如把 "01_0014.npy" 轉成 "01_014"
        # 檔名格式通常是 "Scene_Clip.npy"
        name_part = os.path.splitext(clip_name)[0] # "01_0014"
        
        # 生成這影片每一幀的對照字串
        for frame_idx in range(num_frames):
            # 格式: "影片ID,幀號" -> "01_0014,0" ... "01_0014,150"
            mapping_list.append(f"{name_part},{frame_idx}")

    # 4. 儲存
    save_path = os.path.join(base_dir, 'vlm_fusion', 'data_cache', 'file_mapping.npy')
    np.save(save_path, np.array(mapping_list))
    
    print("-" * 50)
    print(f"✅ 對照表建立完成！")
    print(f"總幀數: {len(mapping_list)}")
    print(f"儲存於: {save_path}")
    print(f"樣本: {mapping_list[:3]} ...")
    
    # 5. 驗證長度
    try:
        score_path = os.path.join(base_dir, 'vlm_fusion', 'data_cache', 'stg_scores.npy')
        scores = np.load(score_path)
        print(f"📏 長度比對: Mapping({len(mapping_list)}) vs Scores({len(scores)})")
        
        if len(mapping_list) == len(scores):
            print("✨ 完美匹配！現在我們可以進行 VLM 圖片讀取了。")
        else:
            diff = len(mapping_list) - len(scores)
            print(f"⚠️ 警告: 還是有 {diff} 幀的差異。")
            print("這可能是因為 scoring_utils 裡面的 smooth_scores 或 clip 處理有些微不同。")
            print("但如果差異很小，我們可以用截斷/補零的方式處理。")
    except Exception as e:
        print(f"無法比對分數長度: {e}")

if __name__ == "__main__":
    extract_mapping_from_gt()