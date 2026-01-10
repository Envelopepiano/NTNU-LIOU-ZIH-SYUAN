# ==========================================
# File: vlm_fusion/train_fusion.py
# Deterministic version (repeatable training)
# ==========================================
#cuda 環境變數
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from fusion_model import CorrectionNet


# ---------------------------
# 1) Deterministic / Seed
# ---------------------------
SEED = 42

def set_deterministic(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch deterministic algorithms (may warn if some ops not supported)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_deterministic(SEED)


# ---------------------------
# 2) Load data (robust path)
# ---------------------------
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    print(f"📂 正在讀取數據，路徑: {cache_dir}")

    stg = np.load(os.path.join(cache_dir, "stg_scores.npy")).astype(np.float32)
    vlm_raw = np.load(os.path.join(cache_dir, "vlm_scores.npy")).astype(np.float32)
    gt  = np.load(os.path.join(cache_dir, "gt_labels.npy")).astype(np.float32)
    if len(stg) < 2000:
        raise RuntimeError("讀取到的樣本數過少，請確認 step1_export_real_scores.py 已正確匯出 40k+。")

    # ✅ 讓 motion「固定不變」：如果沒有真 motion，就用全 0（最穩、可重現）
    motion_path = os.path.join(cache_dir, "motion_scores.npy")
    if os.path.exists(motion_path):
        motion = np.load(motion_path).astype(np.float32)
        if motion.ndim == 1:
            motion = motion.reshape(-1, 1)
    else:
        motion = np.zeros((len(stg), 1), dtype=np.float32)
        np.save(motion_path, motion)
        print(f"ℹ️ motion_scores.npy 不存在，已建立全 0 motion 並存到: {motion_path}")

    # shape to (N,1)
    stg = stg.reshape(-1, 1)
    vlm_raw = vlm_raw.reshape(-1, 1)
    gt  = gt.reshape(-1, 1)

    # ✅ mask: 有跑過 VLM 的 frame（NaN = 未跑過）
    vlm_mask = np.isfinite(vlm_raw).astype(np.float32)  # (N,1)
    # ✅ 餵進模型前，把 NaN 補成 0（避免 NaN 傳染整個 forward）
    vlm = np.nan_to_num(vlm_raw, nan=0.0).astype(np.float32)

    print(f"📊 Samples: {len(stg)}")
    print(f"📊 VLM covered: {int(vlm_mask.sum())} ({float(vlm_mask.mean())*100:.2f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stg_t = torch.from_numpy(stg).to(device)
    vlm_t = torch.from_numpy(vlm).to(device)
    mot_t = torch.from_numpy(motion).to(device)
    gt_t  = torch.from_numpy(gt).to(device)
    msk_t = torch.from_numpy(vlm_mask).to(device)

    return stg_t, vlm_t, mot_t, gt_t, msk_t, device


def train():
    stg, vlm, motion, gt, vlm_mask, device = load_data()

    # ✅ 固定 train/val split（可重現）
    N = gt.shape[0]
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N)
    n_train = int(N * 0.8)
    tr_idx = torch.from_numpy(perm[:n_train]).long().to(device)
    va_idx = torch.from_numpy(perm[n_train:]).long().to(device)

    model = CorrectionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    print(f"🚀 Start training (deterministic) | device={device} | train={len(tr_idx)} val={len(va_idx)}")

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # 只在「有 VLM 的樣本」上訓練更合理（否則 VLM=0 代表缺失會污染訓練）
        tr_has_vlm = (vlm_mask[tr_idx].squeeze(1) > 0.5)
        idx_use = tr_idx[tr_has_vlm]

        if idx_use.numel() == 0:
            raise RuntimeError("Train split 裡沒有任何有 VLM 的樣本。請先跑 step2 增加 VLM 覆蓋率。")

        preds = model(stg[idx_use], vlm[idx_use], motion[idx_use])
        loss = criterion(preds, gt[idx_use])

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                # 評估：對沒有 VLM 的地方，直接用 STG 當分數（不亂猜）
                pred_all = stg.clone()  # baseline
                has_vlm_all = (vlm_mask.squeeze(1) > 0.5)
                pred_all[has_vlm_all] = model(stg[has_vlm_all], vlm[has_vlm_all], motion[has_vlm_all])

                y_true = gt[va_idx].detach().cpu().numpy().ravel()
                y_score = pred_all[va_idx].detach().cpu().numpy().ravel()
                auc_val = roc_auc_score(y_true, y_score)

                print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_auc={auc_val:.4f}")

    # Final report
    model.eval()
    with torch.no_grad():
        pred_all = stg.clone()
        has_vlm_all = (vlm_mask.squeeze(1) > 0.5)
        pred_all[has_vlm_all] = model(stg[has_vlm_all], vlm[has_vlm_all], motion[has_vlm_all])

        y_true_all = gt.detach().cpu().numpy().ravel()
        y_score_all = pred_all.detach().cpu().numpy().ravel()
        y_stg_all = stg.detach().cpu().numpy().ravel()

        auc_fused = roc_auc_score(y_true_all, y_score_all)
        auc_stg = roc_auc_score(y_true_all, y_stg_all)

    print("\n" + "=" * 60)
    print("✅ Done (deterministic)")
    print(f"STG baseline AUC: {auc_stg:.4f}")
    print(f"Fused AUC       : {auc_fused:.4f}")
    print(f"VLM covered     : {int(vlm_mask.sum().item())} / {N}")
    print("=" * 60)


if __name__ == "__main__":
    train()
