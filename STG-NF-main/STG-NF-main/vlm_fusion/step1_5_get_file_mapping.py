# =============================================================================
# File: vlm_fusion/step1_5_get_file_mapping.py
# Description: 
#   修正版：正確使用 init_args 和 PoseSegDataset。
#   功能：找出那 40,791 個分數分別對應 "哪一張圖片"。
# =============================================================================

import sys
import os
import numpy as np

# 1. 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__)) # vlm_fusion/
parent_dir = os.path.dirname(current_dir)                # STG-NF-main/
sys.path.append(parent_dir)

# 2. 修正 Import
try:
    from args import init_args  # <--- 修正這裡：改成 init_args
    from dataset import PoseSegDataset # <--- 修正這裡：直接 import 類別
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def extract_mapping():
    print("🚀 正在解析數據集結構，建立 '分數 <-> 圖片' 對照表...")
    
    # 3. 模擬命令列參數 (為了讓 init_args 正常運作)
    # 這裡我們模擬 'test' 模式的參數
    sys.argv = [
        "step1_5.py", 
        "--dataset", "ShanghaiTech",
        "--only_test",
        # 如果你的資料不在預設的 data/ 下，可能需要加 --data_dir ...
        # 但既然你之前跑得動，這裡應該用預設值即可
    ]
    
    # 4. 初始化參數
    args, model_args = init_args()
    
    print(f"📂 讀取資料路徑: {args.pose_path['test']}")
    
    # 5. 初始化 Dataset
    # 這裡我們要手動建立 Dataset 物件，參數參照 dataset.py 的 __init__
    dataset = PoseSegDataset(
        path_to_json_dir=args.pose_path['test'],
        path_to_vid_dir=args.vid_path['test'],
        dataset=args.dataset,
        evaluate=True, # 測試模式
        seg_len=args.seg_len,
        seg_stride=1, # 測試時 stride 通常為 1
        headless=args.headless,
        scale=args.norm_scale,
        scale_proportional=args.prop_norm_scale,
        train_seg_conf_th=args.train_seg_conf_th
    )
    
    print(f"📊 Dataset 建立成功！長度: {len(dataset)}")
    
    # 6. 檢查與匯出 Metadata
    # dataset.metadata 應該包含了每一筆資料的資訊
    if hasattr(dataset, 'metadata'):
        meta = dataset.metadata
        print(f"🔍 Metadata 範例 (第一筆): {meta[0]}")
        
        # 儲存整個 metadata array，之後我們再解析它
        save_path = os.path.join(current_dir, 'data_cache', 'file_mapping.npy')
        
        # 確保 data_cache 存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.save(save_path, np.array(meta, dtype=object))
        
        print(f"✅ 對照表已建立！")
        print(f"💾 儲存於: {save_path}")
        
        # 簡單檢查長度是否吻合
        try:
            scores = np.load(os.path.join(current_dir, 'data_cache', 'stg_scores.npy'))
            print(f"📏 檢查長度一致性: Dataset({len(meta)}) vs Scores({len(scores)})")
            if len(meta) == len(scores):
                print("✨ 完美！長度完全一致。")
            else:
                print("⚠️ 警告：長度不一致，可能需要檢查參數設定 (如 headless 或 augmentations)。")
        except:
            pass
            
    else:
        print("❌ 錯誤：Dataset 物件中找不到 'metadata' 屬性。")

if __name__ == "__main__":
    extract_mapping()