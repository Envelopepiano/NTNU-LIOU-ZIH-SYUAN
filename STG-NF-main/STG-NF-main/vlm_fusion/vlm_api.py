# ==========================================
# Date: 2026-01-08
# Author: Gemini (Assisted)
# Description: Handles VLM API calls (GPT-4o/Gemini) for semantic anomaly detection.
# ==========================================

import os
import cv2
import json
import random
import time

class VLMAnalyzer:
    def __init__(self, api_key="YOUR_API_KEY"):
        self.api_key = api_key
        # 定義 ShanghaiTech 數據集路徑
        self.dataset_root = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech"

    def get_image_path(self, video_id, frame_id):
        """
        根據 ShanghaiTech 的結構組合圖片路徑
        假設結構: testing/frames/01_001/001.jpg (需根據實際結構微調)
        """
        # 範例路徑，請根據你實際解壓縮後的資料夾名稱修改
        # video_id 例如 "01_014", frame_id 例如 120
        img_name = f"{frame_id:03d}.jpg" 
        path = os.path.join(self.dataset_root, "testing", "frames", video_id, img_name)
        return path

    def check_needs_vlm(self, stg_score, threshold=0.6):
        """
        觸發機制：只在 STG-NF 分數 '不高不低' 或 '疑似異常' 時呼叫 API
        stg_score: 0~1 之間的正規化分數 (假設 1 是異常)
        """
        # 邏輯：如果分數 > 0.6 (有點像異常)，就讓 VLM 確認
        return stg_score > threshold

    def call_vlm_api(self, image_path):
        """
        模擬呼叫 GPT-4o / Gemini Vision API
        真實實作時，這裡要換成 `requests.post` 或 `openai.ChatCompletion`
        """
        if not os.path.exists(image_path):
            print(f"[Warning] Image not found: {image_path}")
            return 0.0, "Image missing"

        # --- 模擬 API 回傳 (Mock) ---
        # 實際使用請替換為真實 API code
        # print(f"Processing {image_path} with VLM...")
        
        # 模擬：隨機產生一個分數與理由 (之後你要接真實 API)
        mock_score = random.uniform(0, 1) 
        mock_reason = "Detected a person running fast." if mock_score > 0.8 else "Normal scene."
        
        return mock_score, mock_reason

    def process_frame(self, video_id, frame_id, stg_score):
        """
        主處理函數
        """
        # 1. 判斷是否需要 VLM (節省成本)
        if not self.check_needs_vlm(stg_score):
            return 0.0, "Skipped (Low STG score)"

        # 2. 取得圖片路徑
        img_path = self.get_image_path(video_id, frame_id)

        # 3. 呼叫 API
        vlm_score, reason = self.call_vlm_api(img_path)
        
        return vlm_score, reason