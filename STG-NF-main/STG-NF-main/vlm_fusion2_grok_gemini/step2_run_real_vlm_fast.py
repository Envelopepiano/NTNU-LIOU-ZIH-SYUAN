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
API_KEY = "XXX"  # <--- ⚠️ 請填入 Key
DATASET_ROOT = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech/testing/frames"

USE_REAL_API = True
API_COST_LIMIT = 500      # 建議跑 500 張以上才有統計意義
MODEL_NAME = "gpt-4o-mini"
MAX_WORKERS = 10          
# ======================================================

def select_samples_smartly(stg_scores, vlm_scores, budget=500):
    """
    智能採樣策略 (Novelty: Uncertainty-based Sampling)
    不只挑分數最高的，要挑「STG 覺得有點異常但又不確定」的樣本
    """
    num_samples = len(stg_scores)
    indices = np.arange(num_samples)
    
    # 1. 正規化 STG 到 0~1
    # 假設 STG 原本是負數 (Log-likelihood)，越小越異常
    # 我們翻轉並正規化：0 (正常) -> 1 (異常)
    stg_min, stg_max = stg_scores.min(), stg_scores.max()
    if stg_max - stg_min > 0:
        stg_norm = (stg_scores - stg_min) / (stg_max - stg_min)
    else:
        stg_norm = np.zeros_like(stg_scores)
    
    # 2. 定義三個區域
    # High Confidence Normal: < 0.5
    # Uncertain / Hard Samples: 0.5 ~ 0.9 (STG 容易誤判的區域，例如腳踏車)
    # High Confidence Anomaly: > 0.9
    
    # 策略分配 Budget
    n_hard = int(budget * 0.7)  # 70% 預算給困難樣本 (最重要！)
    n_top = int(budget * 0.2)   # 20% 給極端異常 (確認用)
    n_norm = budget - n_hard - n_top  # 10% 給正常樣本 (校正用)
    
    # 只考慮還沒跑過 VLM 的樣本
    unprocessed_mask = (vlm_scores == 0.0)
    
    # A. 困難樣本 (Hard Samples): 分數在 0.5 ~ 0.9 之間
    mask_hard = (stg_norm > 0.5) & (stg_norm < 0.9) & unprocessed_mask
    pool_hard = indices[mask_hard]
    
    # B. 極端異常 (Top Anomalies): 分數 >= 0.9
    mask_top = (stg_norm >= 0.9) & unprocessed_mask
    pool_top = indices[mask_top]
    
    # C. 正常樣本 (Normal): 分數 < 0.3
    mask_norm = (stg_norm < 0.3) & unprocessed_mask
    pool_norm = indices[mask_norm]
    
    # 隨機抽取 (無放回)
    selected = []
    actual_hard = min(n_hard, len(pool_hard))
    actual_top = min(n_top, len(pool_top))
    actual_norm = min(n_norm, len(pool_norm))
    
    if len(pool_hard) > 0:
        selected.extend(np.random.choice(pool_hard, actual_hard, replace=False))
    if len(pool_top) > 0:
        selected.extend(np.random.choice(pool_top, actual_top, replace=False))
    if len(pool_norm) > 0:
        selected.extend(np.random.choice(pool_norm, actual_norm, replace=False))
        
    selected = np.array(list(set(selected)))  # 去重
    print(f"🎯 Smart Sampling: {len(selected)} frames (Hard:{actual_hard}, Top:{actual_top}, Norm:{actual_norm})")
    
    return selected

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
    if current_vlm_score != 0.0:
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
            "Output JSON ONLY: {'anomaly_score': float, 'reason': string}."
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
        
        # 讀取或初始化
        if os.path.exists(save_path):
            vlm_scores = np.load(save_path)
        else:
            vlm_scores = np.zeros_like(stg_scores)
    except FileNotFoundError:
        print("❌ 找不到 data_cache")
        return

    # === 使用智能採樣策略 (Uncertainty-based Sampling) ===
    # 不只挑分數最高的，要挑「STG 覺得有點異常但又不確定」的樣本
    target_indices = select_samples_smartly(stg_scores, vlm_scores, budget=API_COST_LIMIT)
    
    tasks = []
    client = OpenAI(api_key=API_KEY)
    
    for i in target_indices:
        if vlm_scores[i] == 0.0:  # 沒跑過才跑
            tasks.append((i, stg_scores[i], mapping[i], vlm_scores[i], client))
    
    print(f"📋 準備處理 {len(tasks)} 張圖片 (Smart Sampling)...")

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
                    
                    # 加入 Log 列表
                    results_log.append({
                        "id": idx,
                        "video": result['video_id'],
                        "frame": result['frame_name'],
                        "stg_score": result['stg_score'],
                        "vlm_score": score,
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