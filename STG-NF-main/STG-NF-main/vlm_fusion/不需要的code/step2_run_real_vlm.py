# =============================================================================
# File: vlm_fusion/step2_run_real_vlm.py
# Description: 真實 VLM 整合版本 (GPT-4o-mini) - Cost Optimized
# =============================================================================

import os
import numpy as np
import base64
import time
import json
from tqdm import tqdm
from openai import OpenAI

# ================= 設定區 (USER SETUP) =================
# 1. 填入你的 API KEY
API_KEY = "XXX"  # <--- 請務必填入你的 Key

# 2. 設定資料集圖片根目錄
DATASET_ROOT = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech/testing/frames"

# 3. 控制參數
USE_REAL_API = True       # True = 真的會扣錢 (但在 Mini 模式下很便宜)
API_COST_LIMIT = 20       # 先跑 20 張試水溫，確認成功後可改為 1000 或 None (跑全量)
MODEL_NAME = "gpt-4o-mini" # 使用最划算的 Mini 模型
# ======================================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_correct_path(base_root, video_id, frame_name):
    """
    修復路徑問題：將 GT 的 '06_0144' 映射到真實資料夾 '06_014'
    """
    # 組合 1: 直接拼接 (如果資料夾名是 4 位數)
    path1 = os.path.join(base_root, video_id, frame_name)
    if os.path.exists(path1): return path1
    
    # 組合 2: 去掉 Video ID 的最後一位 (0144 -> 014)
    # ShanghaiTech 的常見雷點：GT 檔名多補了一個位數
    if '_' in video_id:
        scene, clip = video_id.split('_')
        if len(clip) == 4:
            new_clip = clip[:-1] # 移除最後一個字元
            new_video_id = f"{scene}_{new_clip}"
            path2 = os.path.join(base_root, new_video_id, frame_name)
            if os.path.exists(path2): return path2

    return None

def real_api_call(client, image_path):
    if not os.path.exists(image_path):
        return 0.0, "Image Not Found"

    try:
        base64_image = encode_image(image_path)
        
        # 定義 Prompt：告訴 VLM 我們在找什麼 (ShanghaiTech 的異常定義)
        prompt_text = (
            "You are a surveillance anomaly detector. Analyze the image."
            "Check for: 1. Violence (fighting, chasing). 2. Prohibited vehicles (bike, car, skate) in pedestrian zone. "
            "3. Sudden running or falling."
            "Return JSON: {'anomaly_score': float 0.0 to 1.0, 'reason': string}. "
            "0.0=Normal, 1.0=Anomaly."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low" # <--- 關鍵！強制使用低解析度模式 (85 tokens)，超省錢
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        return result_json.get('anomaly_score', 0.0), result_json.get('reason', 'No reason')

    except Exception as e:
        print(f"\n[API Error] {e}")
        return 0.0, "API Error"

def run_real_vlm():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, 'data_cache')
    
    print(f"🚀 [Step 2] 啟動 VLM ({MODEL_NAME}) | 模式: detail='low'")

    # 1. 讀取資料
    try:
        stg_scores = np.load(os.path.join(cache_dir, 'stg_scores.npy'))
        mapping = np.load(os.path.join(cache_dir, 'file_mapping.npy'))
        save_path = os.path.join(cache_dir, 'vlm_scores.npy')
        
        # 讀取既有進度 (斷點續傳)
        if os.path.exists(save_path):
            vlm_scores = np.load(save_path)
            print(f"📂 讀取舊存檔，目前已處理 {np.count_nonzero(vlm_scores)} 筆")
        else:
            vlm_scores = np.zeros_like(stg_scores)
            
    except FileNotFoundError:
        print("❌ 找不到 data_cache，請先執行 step1")
        return

    # 2. 設定門檻 (針對 STG-NF 的負數分數)
    # 我們只檢查分數最低 (最異常) 的前 5%
    percentile_th = np.percentile(stg_scores, 5) 
    
    print(f"📊 分數統計: Min={stg_scores.min():.2f}, Max={stg_scores.max():.2f}")
    print(f"🎯 異常門檻 (Bottom 5%): {percentile_th:.4f}")
    print(f"💡 只有分數低於 {percentile_th:.4f} 的圖片會被送去檢查")

    client = OpenAI(api_key=API_KEY) if USE_REAL_API else None
    count_run = 0
    count_skipped = 0
    
    # 3. 執行迴圈
    for i in tqdm(range(len(stg_scores))):
        
        # 限制數量
        if API_COST_LIMIT and count_run >= API_COST_LIMIT:
            print(f"\n🛑 達到測試上限 ({API_COST_LIMIT} 張)，暫停執行。")
            break
            
        score = stg_scores[i]
        
        # 邏輯：(分數夠低) AND (之前沒跑過)
        if score < percentile_th and vlm_scores[i] == 0.0:
            
            map_info = mapping[i] # 例如 "06_0144,61"
            video_id, frame_idx = map_info.split(',')
            frame_name = f"{int(frame_idx):03d}.jpg" # 轉成 061.jpg
            
            # 自動找圖 (修復路徑)
            img_path = find_correct_path(DATASET_ROOT, video_id, frame_name)
            
            if img_path:
                if USE_REAL_API:
                    v_score, reason = real_api_call(client, img_path)
                    # gpt-4o-mini 速度快，稍微睡 0.1 秒即可
                    time.sleep(0.1) 
                else:
                    v_score, reason = 0.9, "Mock Test Mode"
                
                vlm_scores[i] = v_score
                count_run += 1
                
                # 印出 Log 讓你安心
                print(f"\n[檢測] ID:{i} 檔名:{video_id}/{frame_name}")
                print(f"      STG分數: {score.item():.2f} (異常) -> VLM觀點: {v_score:.2f} ({reason})")
                
                # 每 10 張存一次檔
                if count_run % 10 == 0: np.save(save_path, vlm_scores)
            else:
                # 找不到圖，不扣次數，但印出警告
                if count_skipped < 5: # 只印前 5 個錯誤避免洗版
                    print(f"\n⚠️ 找不到圖 (Skipped): {video_id}/{frame_name}")
                count_skipped += 1
                
    np.save(save_path, vlm_scores)
    print("\n" + "="*50)
    print(f"✅ 完成！本次執行 VLM: {count_run} 張")
    print(f"❌ 找不到路徑: {count_skipped} 張")
    print(f"💾 結果已儲存: {save_path}")
    print("="*50)

if __name__ == "__main__":
    run_real_vlm()