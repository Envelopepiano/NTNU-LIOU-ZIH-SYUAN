# vlm_fusion/step2_run_vlm.py
import numpy as np
import os
import time
from vlm_api import VLMAnalyzer  # 這是你之前建立的 vlm_api.py

def run_vlm_process():
    print("步驟二：啟動 VLM 語義分析流程...")
    
    # 1. 讀取 STG-NF 分數
    try:
        stg_scores = np.load("data_cache/stg_scores.npy")
    except FileNotFoundError:
        print("錯誤：找不到 data_cache/stg_scores.npy，請先執行步驟一！")
        return

    num_samples = len(stg_scores)
    vlm_scores = np.zeros((num_samples, 1), dtype=np.float32)
    
    # 初始化你的 API 工具
    analyzer = VLMAnalyzer()
    
    print(f"開始掃描 {num_samples} 個樣本...")
    
    # 2. 迴圈處理
    count_api_calls = 0
    for i in range(num_samples):
        current_score = stg_scores[i][0]
        
        # 判斷是否需要 VLM 介入 (這裡設閾值 0.7)
        if analyzer.check_needs_vlm(current_score, threshold=0.7):
            # 這裡我們傳入假的 video_id 和 frame_id 作為測試
            # 真實情況你要傳入對應的影片檔名
            score, reason = analyzer.process_frame("test_video", i, current_score)
            vlm_scores[i] = score
            count_api_calls += 1
            
            # 為了避免刷屏，只印出前幾個
            if count_api_calls <= 5:
                print(f"Frame {i}: STG高分({current_score:.2f}) -> VLM介入 -> 結果:{score:.2f} ({reason})")
        else:
            # 沒觸發 VLM，分數維持 0
            vlm_scores[i] = 0.0
            
    # 儲存 VLM 結果
    np.save("data_cache/vlm_scores.npy", vlm_scores)
    
    print("-" * 30)
    print(f"處理完成！共呼叫 API {count_api_calls} 次 (節省了 {num_samples - count_api_calls} 次)")
    print("結果已儲存至 data_cache/vlm_scores.npy")

if __name__ == "__main__":
    run_vlm_process()
    