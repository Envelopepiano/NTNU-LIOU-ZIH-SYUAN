#!/usr/bin/env python
"""Extract frames from ShanghaiTech training videos."""

import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame with 6-digit format (as expected by gen_patches.py for training)
        frame_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    return frame_count

def main():
    videos_dir = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech/training/videos"
    frames_dir = "/home/user11/NTNU-LIOU,ZIH-SYUAN/data/ShanghaiTech_original/shanghaitech/training/frames"
    
    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.avi')])
    print(f"Found {len(video_files)} videos to process")
    
    for video_file in tqdm(video_files, desc="Extracting frames"):
        # Extract video name without extension (e.g., "01_001" from "01_001.avi")
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(videos_dir, video_file)
        output_dir = os.path.join(frames_dir, video_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Skip if already extracted
        if len(os.listdir(output_dir)) > 0:
            continue
        
        frame_count = extract_frames(video_path, output_dir)
        # print(f"Extracted {frame_count} frames from {video_name}")

if __name__ == "__main__":
    main()
