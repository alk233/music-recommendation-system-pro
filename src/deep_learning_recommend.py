# 深度学习推荐模块
import os
import sys
from copy import deepcopy
from typing import Optional, Dict, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_FILE, NCF_MODEL_PATH, NCF_SAMPLE_SIZE, USERS_FILE, HISTORY_FILE


def _make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


# NCF模型
class NCF(nn.Module):
    # 神经协同过滤模型（轻量 MLP + Dropout）
    def __init__(self, num_users, num_songs, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.song_emb = nn.Embedding(num_songs, emb_dim)
        hidden = max(64, emb_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.song_emb.weight, std=0.01)

    def forward(self, user, song):
        u = self.user_emb(user)
        s = self.song_emb(song)
        x = torch.cat([u, s], dim=1)
        return self.mlp(x).squeeze(-1)


def _build_user_sampling_tables(df: pd.DataFrame, weighted_pos: bool = False):
    if df is None or df.empty:
        return {}, {}, {}
    cols = ["user", "song"] + (["play_count"] if "play_count" in df.columns else [])
    work = df[cols].copy()
    if "play_count" not in work.columns:
        work["play_count"] = 1.0
    work["play_count"] = pd.to_numeric(work["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    merged = work.groupby(["user", "song"], as_index=False)["play_count"].max()
    user_pos: Dict[int, Set[int]] = {}
    pos_arrays = {}
    pos_probs = {}
    for uid, sub in merged.groupby("user", sort=False):
        uid = int(uid)
        songs = sub["song"].astype(np.int64).values
        user_pos[uid] = set(int(x) for x in songs.tolist())
        pos_arrays[uid] = songs
        if weighted_pos:
            w = np.power(sub["play_count"].astype(np.float64).values, 0.75)
            sw = float(np.sum(w))
            if sw > 1e-12:
                pos_probs[uid] = w / sw
            else:
                pos_probs[uid] = np.full(len(songs), 1.0 / max(1, len(songs)), dtype=np.float64)
        else:
            pos_probs[uid] = None
    return user_pos, pos_arrays, pos_probs


def _build_song_sampling_probs(df: pd.DataFrame, num_songs: int) -> Optional[np.ndarray]:
    if df is None or df.empty or num_songs <= 0:
        return None
    sids = pd.to_numeric(df["song"], errors="coerce").dropna().astype(np.int64).values
    if sids.size == 0:
        return None
    cnt = np.bincount(sids, minlength=num_songs).astype(np.float64)
    smoothed = np.power(np.maximum(cnt, 1.0), 0.75)
    total = float(np.sum(smoothed))
    if total <= 1e-12:
        return None
    return smoothed / total


def _sample_bpr_triplets(
    user_pos: Dict[int, Set[int]],
    all_users: np.ndarray,
    num_songs: int,
    batch_size: int,
    rng: np.random.Generator,
    pos_arrays: Optional[dict] = None,
    pos_probs: Optional[dict] = None,
    song_sampling_probs: Optional[np.ndarray] = None,
    song_ids: Optional[np.ndarray] = None,
    neg_k: int = 1,
    popular_negative_ratio: float = 0.67,
    max_neg_retry: int = 20,
):
    users = rng.choice(all_users, size=batch_size, replace=True)
    if pos_arrays is None:
        pos_arrays = {int(u): np.asarray(list(pos_set), dtype=np.int64) for u, pos_set in user_pos.items()}
    if song_sampling_probs is not None:
        if song_ids is None or len(song_ids) != len(song_sampling_probs):
            song_ids = np.arange(num_songs, dtype=np.int64)
        sampled_negs_pop = rng.choice(song_ids, size=(batch_size, neg_k), replace=True, p=song_sampling_probs)
    else:
        sampled_negs_pop = rng.integers(0, num_songs, size=(batch_size, neg_k), dtype=np.int64)
    sampled_negs_rand = rng.integers(0, num_songs, size=(batch_size, neg_k), dtype=np.int64)
    use_pop_mask = rng.random(size=(batch_size, neg_k)) < float(np.clip(popular_negative_ratio, 0.0, 1.0))
    neg_items = np.where(use_pop_mask, sampled_negs_pop, sampled_negs_rand).astype(np.int64)

    pos_items = np.empty(batch_size, dtype=np.int64)
    for i, u in enumerate(users.astype(np.int64)):
        uid = int(u)
        pos_set = user_pos[uid]
        arr = pos_arrays.get(uid)
        if arr is None or len(arr) == 0:
            arr = np.asarray(list(pos_set), dtype=np.int64)
            pos_arrays[uid] = arr
        prob = None if pos_probs is None else pos_probs.get(uid)
        if prob is not None and len(prob) == len(arr):
            pos = int(rng.choice(arr, p=prob))
        else:
            pos = int(rng.choice(arr))
        pos_items[i] = pos

        for j in range(neg_k):
            neg = int(neg_items[i, j])
            retry = 0
            while neg in pos_set and retry < max_neg_retry:
                neg = int(rng.integers(0, num_songs))
                retry += 1
            if neg in pos_set:
                while True:
                    neg = int(rng.integers(0, num_songs))
                    if neg not in pos_set:
                        break
            neg_items[i, j] = neg
    if int(neg_k) == 1:
        return users.astype(np.int64), pos_items, neg_items[:, 0]
    return users.astype(np.int64), pos_items, neg_items


def _estimate_pairwise_acc(
    model: nn.Module,
    user_pos: Dict[int, Set[int]],
    all_users: np.ndarray,
    num_songs: int,
    rng: np.random.Generator,
    device: torch.device,
    n_check: int,
    pos_arrays: Optional[dict] = None,
) -> float:
    if len(all_users) == 0:
        return 0.0
    u_np, p_np, n_np = _sample_bpr_triplets(
        user_pos=user_pos,
        all_users=all_users,
        num_songs=num_songs,
        batch_size=max(1, int(n_check)),
        rng=rng,
        pos_arrays=pos_arrays,
    )
    with torch.no_grad():
        u_t = torch.tensor(u_np, dtype=torch.long, device=device)
        p_t = torch.tensor(p_np, dtype=torch.long, device=device)
        n_t = torch.tensor(n_np, dtype=torch.long, device=device)
        wins = (model(u_t, p_t) - model(u_t, n_t) > 0).float().mean().item()
    return float(wins)


def _load_registered_history_interactions():
    """读取 users.json + user_history.json，返回 (user, song, play_count) 交互。"""
    if not (os.path.isfile(USERS_FILE) and os.path.isfile(HISTORY_FILE)):
        return pd.DataFrame(columns=["user", "song", "play_count"])
    try:
        import json

        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["user", "song", "play_count"])
    name_to_uid = {}
    for uname, uinfo in (users or {}).items():
        try:
            name_to_uid[str(uname)] = int((uinfo or {}).get("user_id"))
        except Exception:
            continue
    pairs = []
    for uname, songs in (all_history or {}).items():
        uid = name_to_uid.get(str(uname))
        if uid is None:
            continue
        for sid in songs or []:
            try:
                pairs.append((uid, int(sid), 1.0))
            except Exception:
                continue
    if not pairs:
        return pd.DataFrame(columns=["user", "song", "play_count"])
    out = pd.DataFrame(pairs, columns=["user", "song", "play_count"])
    return out.groupby(["user", "song"], as_index=False)["play_count"].max()


def train_ncf_model(
    n_epochs=12,
    batch_size=1024,
    lr=0.0006,
    emb_dim=64,
    neg_k=3,
    popular_negative_ratio=0.67,
    min_epochs=5,
    val_ratio=0.15,
    early_stop_patience=4,
):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"正在读取数据（前{NCF_SAMPLE_SIZE}条）...")
    df = pd.read_csv(
        DATA_FILE,
        nrows=NCF_SAMPLE_SIZE,
        usecols=["user", "song", "play_count"],
        dtype={"user": "int32", "song": "int32", "play_count": "float32"},
    )
    ext = _load_registered_history_interactions()
    if not ext.empty:
        df = pd.concat([df, ext.astype({"user": "int32", "song": "int32", "play_count": "float32"})], ignore_index=True)
    df["play_count"] = pd.to_numeric(df["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    df = df.groupby(["user", "song"], as_index=False)["play_count"].max()

    # 关键优化：稀疏原始 ID -> 密集 ID
    unique_users = np.array(sorted(df["user"].astype(int).unique()), dtype=np.int64)
    unique_songs = np.array(sorted(df["song"].astype(int).unique()), dtype=np.int64)
    user2idx = {int(u): i for i, u in enumerate(unique_users)}
    song2idx = {int(s): i for i, s in enumerate(unique_songs)}
    df = df.copy()
    df["user"] = df["user"].astype(int).map(user2idx).astype(np.int32)
    df["song"] = df["song"].astype(int).map(song2idx).astype(np.int32)
    num_users = len(unique_users)
    num_songs = len(unique_songs)
    print(f"密集编码：{num_users} 用户 × {num_songs} 歌曲")

    train, val = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)
    model = NCF(num_users, num_songs, emb_dim=emb_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=1, min_lr=1e-5
    )
    scaler = _make_grad_scaler(device)

    user_pos, train_pos_arrays, train_pos_probs = _build_user_sampling_tables(train, weighted_pos=True)
    all_users = np.array(list(user_pos.keys()), dtype=np.int64)
    val_user_pos, val_pos_arrays, _ = _build_user_sampling_tables(val, weighted_pos=False)
    val_users = np.array(list(val_user_pos.keys()), dtype=np.int64)
    song_ids = np.arange(num_songs, dtype=np.int64)
    song_sampling_probs = _build_song_sampling_probs(train, num_songs)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(120, len(train) // max(batch_size, 1))
    val_n_check = min(4000, max(1000, len(val_users) * 4)) if len(val_users) > 0 else 0

    best_acc = -1.0
    best_state = None
    no_improve_epochs = 0

    print("开始训练模型...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            u_np, p_np, n_np = _sample_bpr_triplets(
                user_pos=user_pos,
                all_users=all_users,
                num_songs=num_songs,
                batch_size=batch_size,
                rng=rng,
                pos_arrays=train_pos_arrays,
                pos_probs=train_pos_probs,
                song_sampling_probs=song_sampling_probs,
                song_ids=song_ids,
                neg_k=max(1, int(neg_k)),
                popular_negative_ratio=popular_negative_ratio,
            )
            user_t = torch.tensor(u_np, dtype=torch.long, device=device)
            pos_t = torch.tensor(p_np, dtype=torch.long, device=device)
            optimizer.zero_grad()
            with _autocast_ctx(device):
                pos_score = model(user_t, pos_t)
                if n_np.ndim == 1:
                    neg_t = torch.tensor(n_np, dtype=torch.long, device=device)
                    neg_score = model(user_t, neg_t)
                    loss = -F.logsigmoid(pos_score - neg_score).mean()
                else:
                    bsz, kneg = int(n_np.shape[0]), int(n_np.shape[1])
                    neg_t = torch.tensor(n_np.reshape(-1), dtype=torch.long, device=device)
                    user_rep = user_t.repeat_interleave(kneg)
                    neg_score = model(user_rep, neg_t).view(bsz, kneg)
                    pos_rep = pos_score.unsqueeze(1).expand_as(neg_score)
                    loss = -F.logsigmoid(pos_rep - neg_score).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, steps_per_epoch)

        model.eval()
        if val_n_check > 0:
            val_acc = _estimate_pairwise_acc(
                model=model,
                user_pos=val_user_pos,
                all_users=val_users,
                num_songs=num_songs,
                rng=rng,
                device=device,
                n_check=val_n_check,
                pos_arrays=val_pos_arrays,
            )
        else:
            val_acc = 0.0
        scheduler.step(val_acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{n_epochs}, BPR Loss: {avg_loss:.4f}, "
            f"Val Pairwise Acc: {val_acc:.4f}, LR: {cur_lr:.6f}"
        )
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if (epoch + 1) >= max(1, int(min_epochs)) and no_improve_epochs >= max(1, int(early_stop_patience)):
                print(f"触发早停：连续 {no_improve_epochs} 个 epoch 验证集无提升。")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if len(val_users) > 0:
        final_n_check = min(5000, max(1000, len(val_users) * 4))
        final_acc = _estimate_pairwise_acc(
            model=model,
            user_pos=val_user_pos,
            all_users=val_users,
            num_songs=num_songs,
            rng=rng,
            device=device,
            n_check=final_n_check,
            pos_arrays=val_pos_arrays,
        )
        print(f"Final Pairwise Acc: {final_acc:.4f} (best_val={best_acc:.4f})")

    os.makedirs(os.path.dirname(NCF_MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "user2idx": user2idx,
            "song2idx": song2idx,
            "num_users": num_users,
            "num_songs": num_songs,
        },
        NCF_MODEL_PATH,
    )
    print(f"模型已保存为 {NCF_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_ncf_model()
