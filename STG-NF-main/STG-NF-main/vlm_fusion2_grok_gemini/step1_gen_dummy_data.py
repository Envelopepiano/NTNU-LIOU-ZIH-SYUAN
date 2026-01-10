# vlm_fusion/step1_gen_dummy_data.py
import numpy as np
import os

def generate_dummy():
    print("正在產生模擬的 STG-NF 分數與 Ground Truth...")
    
    # 假設有 1000 個測試樣本 (Frames)
    num_samples = 1000
    
    # 1. 模擬 STG-NF 分數 (0~1 之間)
    # 假設大部分是正常的(低分)，少部分是異常(高分)
    stg_scores = np.random.rand(num_samples, 1).astype(np.float32)
    
    # 2. 模擬 Ground Truth (0:正常, 1:異常)
    gt_labels = np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)
    
    # 儲存檔案
    if not os.path.exists("data_cache"):
        os.makedirs("data_cache")
        
    np.save("data_cache/stg_scores.npy", stg_scores)
    np.save("data_cache/gt_labels.npy", gt_labels)
    
    print(f"成功產生模擬數據：")
    print(f" - data_cache/stg_scores.npy {stg_scores.shape}")
    print(f" - data_cache/gt_labels.npy {gt_labels.shape}")

if __name__ == "__main__":
    generate_dummy()