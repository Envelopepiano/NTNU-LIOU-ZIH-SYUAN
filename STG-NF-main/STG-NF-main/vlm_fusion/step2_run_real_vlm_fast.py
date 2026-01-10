# =============================================================================
# File: vlm_fusion/step2_run_real_vlm_fast.py
# Description: V7.0 - 包含 JSON Log 輸出與嚴格評分
# =============================================================================

import os
import numpy as np
import base64
import time
import json
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# ================= 設定區 (USER SETUP) =================
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY and USE_REAL_API:
    raise RuntimeError("OPENAI_API_KEY not set. Please export OPENAI_API_KEY before running.")
DATASET_ROOT = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech/testing/frames"

USE_REAL_API = True
API_COST_LIMIT = 500      # 建議跑 500 張以上才有統計意義
MODEL_NAME = "gpt-4o-mini"
MAX_WORKERS = 10          
# ======================================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_correct_path(base_root, video_id, frame_name):
    path1 = os.path.join(base_root, video_id, frame_name)
    if os.path.exists(path1): return path1
    if '_' in video_id:
        scene, clip = video_id.split('_')
        if len(clip) == 4:
            new_video_id = f"{scene}_{clip[:-1]}"
            path2 = os.path.join(base_root, new_video_id, frame_name)
            if os.path.exists(path2): return path2
    return None

def process_single_frame(args):
    idx, stg_score, map_info, current_vlm_score, client = args
    
    # 已經跑過就跳過
    if np.isfinite(current_vlm_score):
        return None


    video_id, frame_idx = map_info.split(',')
    frame_name = f"{int(frame_idx):03d}.jpg"
    img_path = find_correct_path(DATASET_ROOT, video_id, frame_name)
    
    if not img_path:
        return {'idx': idx, 'status': 'error', 'reason': 'Image Not Found'}

    if not USE_REAL_API:
        return {'idx': idx, 'status': 'success', 'score': 0.95, 'reason': 'Mock Test'}

    try:
        base64_image = encode_image(img_path)
        
        # === V7.0 極致嚴格 Prompt ===
        prompt_text = (
            "Task: Detect anomalies in ShanghaiTech pedestrian zone. "
            "You act as a STRICT FILTER. "
            "FORMULA: Score = Max(Vehicle, Violence, Motion). "
            "\n"
            "1. [VEHICLES] (Bicycles, Cars, Scooters, Skates, Carts): "
            "   - DETECTED = 0.95 (Severe Anomaly). "
            "   - Even if riding slowly or pushing it, it is BANNED. Score MUST be > 0.9."
            "\n"
            "2. [VIOLENCE] (Fighting, Chasing, Robbery...): "
            "   - DETECTED = 0.95."
            "\n"
            "3. [MOTION] (Sudden Running, Falling, Jumping...): "
            "   - DETECTED = 0.7 to 0.9 (Warning)."
            "\n"
            "4. [NORMAL] (Walking, Standing): "
            "   - Score <=0.1."
            "\n"
            'Output JSON ONLY (double quotes): {"anomaly_score": 0.0, "reason": "..."}.'
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
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        
        res = json.loads(response.choices[0].message.content)
        return {
            'idx': idx, 
            'video_id': video_id,
            'frame_name': frame_name,
            'stg_score': float(stg_score),
            'status': 'success', 
            'score': res.get('anomaly_score', 0.0), 
            'reason': res.get('reason', 'None')
        }

    except Exception as e:
        return {'idx': idx, 'status': 'error', 'reason': str(e)}

def run_fast_vlm():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, 'data_cache')
    print(f"🚀 [Step 2 V7] 啟動 VLM ({MODEL_NAME}) + JSON Log")

    try:
        stg_scores = np.load(os.path.join(cache_dir, 'stg_scores.npy'))
        mapping = np.load(os.path.join(cache_dir, 'file_mapping.npy'))
        save_path = os.path.join(cache_dir, 'vlm_scores.npy')
        
        # 讀取或初始化（NaN = 未跑過）
        if os.path.exists(save_path):
            vlm_scores = np.load(save_path).astype(np.float32)
        else:
            vlm_scores = np.full_like(stg_scores, np.nan, dtype=np.float32)

    except FileNotFoundError:
        print("❌ 找不到 data_cache")
        return

    # ===== 修正：現在 STG 分數是 0~1，越大越異常 =====
    # 選擇 STG 認為高度異常的樣本 (Top 5%)
    percentile_th = np.percentile(stg_scores, 95)  # 取 Top 5% (分數最高的)
    print(f"🎯 異常門檻: STG > {percentile_th:.4f} (越大越異常)")
    print(f"📊 STG 分數統計: min={stg_scores.min():.4f}, max={stg_scores.max():.4f}, mean={stg_scores.mean():.4f}")

    # 找出高於門檻的樣本
    high_score_indices = np.where(stg_scores > percentile_th)[0]
    print(f"📌 高於門檻的樣本數: {len(high_score_indices)}")

    tasks = []
    client = OpenAI(api_key=API_KEY)
    
    for i in high_score_indices:
        # 還沒跑過 VLM 的才加入（NaN 才是沒跑過）
        if np.isnan(vlm_scores[i]):
            tasks.append((i, stg_scores[i], mapping[i], vlm_scores[i], client))
            if len(tasks) >= API_COST_LIMIT:
                break
    
    print(f"📋 準備處理 {len(tasks)} 張圖片...")

    results_log = [] # 用來存詳細 log
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(process_single_frame, t): t[0] for t in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks)):
            try:
                result = future.result()
                if result and result['status'] == 'success':
                    idx = result['idx']
                    score = result['score']
                    vlm_scores[idx] = score
                    
                    # 加入 Log 列表 (轉換 numpy 類型為 Python 原生類型)
                    results_log.append({
                        "id": int(idx),  # numpy.int64 -> int
                        "video": result['video_id'],
                        "frame": result['frame_name'],
                        "stg_score": float(result['stg_score']),  # 確保是 float
                        "vlm_score": float(score),  # 確保是 float
                        "reason": result['reason']
                    })
                    
                    # 即時顯示高分
                    if score > 0.8:
                        print(f"✅ {result['video_id']} | VLM:{score} | {result['reason'][:50]}...")

            except Exception as e:
                print(f"Thread Error: {e}")

    # 存檔 1: 分數矩陣
    np.save(save_path, vlm_scores)
    
    # 存檔 2: 詳細 JSON Log (這就是你要的！)
    log_path = os.path.join(cache_dir, 'vlm_results_log.json')
    # 如果有舊的 log，要讀出來合併 (Append mode)
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            old_logs = json.load(f)
            old_logs.extend(results_log)
            results_log = old_logs
            
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"✅ 完成！Log 已儲存至: {log_path}")
    print("你可以打開這個 json 檔查看 GPT 的每一句解釋。")
    print("="*50)

if __name__ == "__main__":
    run_fast_vlm()