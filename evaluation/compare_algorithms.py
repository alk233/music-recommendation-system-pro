#!/usr/bin/env python3
"""
离线对比 UserCF / ItemCF / SVD / 内容推荐 /（可选）NCF / 基线。

协议（与工业界隐式反馈评测常见做法一致）：
  对每个测试样本 (user, 正例歌曲)：
    候选集 = {正例} + 99 首该用户在训练集中未见过的随机负例
    各算法对 100 首歌打分并排序，看正例是否进入 Top-K。

指标在固定协议下可横向对比：HR@K 与 Recall@K 在单正例下数值相同，另有 NDCG@K、Precision@K、MRR、F1@K（K 默认 10）。
融合行另报告「候选AUC」：在 1 正例与固定条数负例组成的候选上，以融合分为分数、用 sklearn 的 ROC AUC 度量正负可分离程度；融合分本身为五路分数在候选内鲁棒校准后与线上一键融合相同的加权方式。

用法（在项目根目录 music-recommendation-system-pro 下执行）：
  python evaluation/compare_algorithms.py
  python evaluation/compare_algorithms.py --max-users 150 --k 10 --with-ncf
"""
from __future__ import annotations

import argparse
import os
import sys
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, KNNBasic, SVD as SurpriseSVD

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_FILE, CF_SAMPLE_SIZE, CF_KNN_K, NCF_MODEL_PATH, NCF_SAMPLE_SIZE

# 与 recommend_utils 一致：避免 Surprise 余弦在零范数向量上除零
_CF_RATING_EPS = 1e-6
_SVD_FACTORS = 160
_SVD_EPOCHS = 36
_SVD_LR_ALL = 0.0025
_SVD_REG_ALL = 0.03
_CF_KNN_MIN_SUPPORT = 2

# 复现性
RNG = np.random.default_rng(42)


def _normalize_idx_map(m):
    if not isinstance(m, dict):
        return {}
    out = {}
    for k, v in m.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _ndcg_at_k(ranked_items: list, positive: int, k: int) -> float:
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == positive:
            dcg = 1.0 / np.log2(rank + 1)
            idcg = 1.0 / np.log2(2)
            return float(dcg / idcg)
    return 0.0


def _hit_at_k(ranked_items: list, positive: int, k: int) -> float:
    return 1.0 if positive in ranked_items[:k] else 0.0


def _precision_at_k(ranked_items: list, positive: int, k: int) -> float:
    """仅 1 个相关项时：进入前 K 则为 1/K，否则 0。"""
    return (1.0 / k) if positive in ranked_items[:k] else 0.0


def _recall_at_k(ranked_items: list, positive: int, k: int) -> float:
    """仅 1 个相关项时与 HR@K 相同。"""
    return _hit_at_k(ranked_items, positive, k)


def _mrr(ranked_items: list, positive: int) -> float:
    """首个相关项位置的倒数排名；未出现为 0。"""
    for rank, item in enumerate(ranked_items, start=1):
        if item == positive:
            return 1.0 / rank
    return 0.0


def _f1_at_k(ranked_items: list, positive: int, k: int) -> float:
    """单正例下由 P@K 与 R@K 计算 F1。"""
    if positive not in ranked_items[:k]:
        return 0.0
    p = 1.0 / k
    r = 1.0
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def load_interactions(nrows: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, nrows=nrows, usecols=["user", "song", "play_count"])
    df = df.copy()
    df["play_count"] = np.maximum(
        df["play_count"].astype(np.float64), _CF_RATING_EPS
    )
    return df


def build_train_test(
    df: pd.DataFrame,
    min_interactions: int,
    test_ratio: float,
    max_test_users: int,
    max_test_samples: int,
):
    """按用户划分：每个用户至少 min_interactions 条；按比例划测试；限制用户与样本数以控制耗时。"""
    cnt = df.groupby("user").size()
    eligible_users = cnt[cnt >= min_interactions].index.tolist()
    if not eligible_users:
        raise ValueError("没有满足最小时长的用户，请减小 min_interactions 或增大 nrows")

    random.shuffle(eligible_users)
    eligible_users = eligible_users[:max_test_users]

    train_rows = []
    test_pairs = []

    for u in eligible_users:
        sub = df[df["user"] == u]
        idx = sub.index.tolist()
        RNG.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx = set(idx[:n_test])
        train_idx = [i for i in idx if i not in test_idx]

        train_rows.append(sub.loc[train_idx])
        for i in idx[:n_test]:
            row = sub.loc[i]
            test_pairs.append((int(row["user"]), int(row["song"])))

    if len(test_pairs) > max_test_samples:
        test_pairs = random.sample(test_pairs, max_test_samples)

    train_df = pd.concat(train_rows, ignore_index=True)
    return train_df, test_pairs


def build_content_encoder(df_meta: pd.DataFrame):
    """与 recommend_utils 中内容特征一致，但不依赖 Streamlit。"""
    content_features = ["artist_name", "title", "release", "year", "artist_familiarity", "artist_hotttnesss"]
    if "release" not in df_meta.columns:
        df_meta = df_meta.copy()
        df_meta["release"] = "unknown"
    songs = df_meta.drop_duplicates("song")[["song"] + content_features].set_index("song")
    artist_hasher = FeatureHasher(n_features=32, input_type="string")
    title_hasher = FeatureHasher(n_features=64, input_type="string")
    release_hasher = FeatureHasher(n_features=32, input_type="string")
    scaler = MinMaxScaler()
    artist_names = songs["artist_name"].fillna("unknown").astype(str).values
    title_names = songs["title"].fillna("unknown").astype(str).values
    release_names = songs["release"].fillna("unknown").astype(str).values
    num_features = songs[["year", "artist_familiarity", "artist_hotttnesss"]].fillna(0).values
    artist_list = [[name] for name in artist_names]
    title_list = [[name] for name in title_names]
    release_list = [[name] for name in release_names]
    artist_hasher.fit(artist_list)
    title_hasher.fit(title_list)
    release_hasher.fit(release_list)
    scaler.fit(num_features)
    return artist_hasher, title_hasher, release_hasher, scaler, songs


def song_vector(song_id: int, artist_hasher, title_hasher, release_hasher, scaler, songs) -> np.ndarray | None:
    if song_id not in songs.index:
        return None
    row = songs.loc[song_id]
    artist_name = str(row["artist_name"]) if pd.notna(row["artist_name"]) else "unknown"
    title_name = str(row["title"]) if pd.notna(row["title"]) else "unknown"
    release_name = str(row["release"]) if pd.notna(row["release"]) else "unknown"
    artist_feature = artist_hasher.transform([[artist_name]]).toarray()[0]
    title_feature = title_hasher.transform([[title_name]]).toarray()[0]
    release_feature = release_hasher.transform([[release_name]]).toarray()[0]
    num_values = [
        [
            row["year"] if pd.notna(row["year"]) else 0,
            row["artist_familiarity"] if pd.notna(row["artist_familiarity"]) else 0,
            row["artist_hotttnesss"] if pd.notna(row["artist_hotttnesss"]) else 0,
        ]
    ]
    num_feature = scaler.transform(num_values)[0]
    return np.hstack([artist_feature, title_feature, release_feature, num_feature])


def user_content_profile(train_songs: list[int], artist_hasher, title_hasher, release_hasher, scaler, songs) -> np.ndarray | None:
    vecs = []
    for sid in train_songs:
        v = song_vector(sid, artist_hasher, title_hasher, release_hasher, scaler, songs)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def _calibrate_scores_list(vals: list) -> list:
    """与线上一致：鲁棒 z（median/IQR）+ sigmoid 到约 0～1，用于单候选集内多路分数校准。"""
    if not vals:
        return []
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr < 1e-8:
        mu = float(np.mean(arr))
        std = float(np.std(arr))
        z = (arr - mu) / (std + 1e-8) if std > 1e-8 else np.zeros_like(arr)
    else:
        z = (arr - med) / (iqr / 1.349 + 1e-8)
    z = np.clip(z, -8.0, 8.0)
    return [float(1.0 / (1.0 + np.exp(-x))) for x in z]


def _normalize_hybrid_weights_five(w: dict | None) -> dict:
    """五路权重归一；兼容旧键 cf（均分到 usercf/itemcf/svd）。"""
    w = dict(w or {})
    out = {k: max(0.0, float(w.get(k, 0.0))) for k in ("usercf", "itemcf", "svd", "ncf", "content")}
    cf = max(0.0, float(w.get("cf", 0.0)))
    if cf > 0:
        out["usercf"] += cf / 3.0
        out["itemcf"] += cf / 3.0
        out["svd"] += cf / 3.0
    s = sum(out.values())
    if s < 1e-12:
        out = {"usercf": 0.20, "itemcf": 0.15, "svd": 0.15, "ncf": 0.40, "content": 0.10}
        s = sum(out.values())
    return {k: out[k] / s for k in out}


def _stretch_for_surprise_knn(s: pd.Series) -> pd.Series:
    v = s.astype(np.float64).values
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return pd.Series(np.full(len(v), 5.5), index=s.index)
    return pd.Series(1.0 + 9.0 * (v - lo) / (hi - lo), index=s.index)


def _stretch_for_surprise_svd(s: pd.Series) -> pd.Series:
    # 与线上 recommend_utils 保持一致：SVD 采用线性拉伸映射
    return _stretch_for_surprise_knn(s)


def fit_surprise_models(train_df: pd.DataFrame, seed: int = 42):
    df_knn = train_df[["user", "song", "play_count"]].copy()
    df_knn["play_count"] = _stretch_for_surprise_knn(df_knn["play_count"])
    df_svd = train_df[["user", "song", "play_count"]].copy()
    df_svd["play_count"] = _stretch_for_surprise_svd(df_svd["play_count"])
    reader = Reader(rating_scale=(1.0, 10.0))
    data_knn = Dataset.load_from_df(df_knn, reader)
    trainset_knn = data_knn.build_full_trainset()
    data_svd = Dataset.load_from_df(df_svd, reader)
    trainset_svd = data_svd.build_full_trainset()

    usercf = KNNBasic(
        k=CF_KNN_K, sim_options={"name": "cosine", "user_based": True, "min_support": _CF_KNN_MIN_SUPPORT}
    )
    usercf.fit(trainset_knn)

    itemcf = KNNBasic(
        k=CF_KNN_K, sim_options={"name": "cosine", "user_based": False, "min_support": _CF_KNN_MIN_SUPPORT}
    )
    itemcf.fit(trainset_knn)

    svd = SurpriseSVD(
        n_factors=_SVD_FACTORS,
        n_epochs=_SVD_EPOCHS,
        lr_all=_SVD_LR_ALL,
        reg_all=_SVD_REG_ALL,
        random_state=seed,
        biased=True,
    )
    svd.fit(trainset_svd)

    return usercf, itemcf, svd, trainset_svd


def train_ncf_in_memory(train_df: pd.DataFrame, seed: int = 42):
    """
    仅内存训练 NCF（用于同规模公平对比），不写入任何模型文件。
    返回: (model, num_users, num_songs, user2idx, song2idx)
    """
    import torch
    import torch.nn.functional as F
    from src.deep_learning_recommend import NCF

    if train_df is None or train_df.empty:
        return None, 0, 0, {}, {}

    work = train_df[["user", "song", "play_count"]].copy()
    work["play_count"] = pd.to_numeric(work["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    work = work.groupby(["user", "song"], as_index=False)["play_count"].max()

    # 与主训练一致：按排序后的唯一 ID 做密集映射
    unique_users = np.array(sorted(work["user"].astype(int).unique()), dtype=np.int64)
    unique_songs = np.array(sorted(work["song"].astype(int).unique()), dtype=np.int64)
    user2idx = {int(u): i for i, u in enumerate(unique_users)}
    song2idx = {int(s): i for i, s in enumerate(unique_songs)}
    work["u_enc"] = work["user"].astype(int).map(user2idx).astype(np.int32)
    work["s_enc"] = work["song"].astype(int).map(song2idx).astype(np.int32)

    num_users = int(len(unique_users))
    num_songs = int(len(unique_songs))
    if num_users <= 0 or num_songs <= 1:
        return None, num_users, num_songs, user2idx, song2idx

    user_pos = defaultdict(set)
    for u, s in zip(work["u_enc"].astype(int).values, work["s_enc"].astype(int).values):
        user_pos[int(u)].add(int(s))
    all_users = np.array(list(user_pos.keys()), dtype=np.int64)
    if len(all_users) == 0:
        return None, num_users, num_songs, user2idx, song2idx

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NCF(num_users, num_songs, emb_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-6)
    rng = np.random.default_rng(seed)

    batch_size = 1024
    n_epochs = 4
    steps_per_epoch = max(40, len(work) // max(1, batch_size))

    model.train()
    for _ in range(n_epochs):
        for _ in range(steps_per_epoch):
            users = rng.choice(all_users, size=batch_size, replace=True).astype(np.int64)
            pos_items = np.empty(batch_size, dtype=np.int64)
            neg_items = np.empty(batch_size, dtype=np.int64)
            for i, u in enumerate(users):
                pos_set = user_pos[int(u)]
                pos = int(rng.choice(np.asarray(list(pos_set), dtype=np.int64)))
                neg = int(rng.integers(0, num_songs))
                retry = 0
                while neg in pos_set and retry < 20:
                    neg = int(rng.integers(0, num_songs))
                    retry += 1
                if neg in pos_set:
                    while True:
                        neg = int(rng.integers(0, num_songs))
                        if neg not in pos_set:
                            break
                pos_items[i] = pos
                neg_items[i] = neg

            u_t = torch.tensor(users, dtype=torch.long, device=device)
            p_t = torch.tensor(pos_items, dtype=torch.long, device=device)
            n_t = torch.tensor(neg_items, dtype=torch.long, device=device)
            pos_score = model(u_t, p_t)
            neg_score = model(u_t, n_t)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    return model, num_users, num_songs, user2idx, song2idx


def popularity_scores(train_df: pd.DataFrame) -> dict[int, float]:
    freq = train_df.groupby("song")["play_count"].sum()
    mx = freq.max() if len(freq) else 1.0
    return {int(s): float(v) / mx for s, v in freq.items()}


def user_train_songs(train_df: pd.DataFrame) -> dict[int, set[int]]:
    d = defaultdict(set)
    for _, r in train_df.iterrows():
        d[int(r["user"])].add(int(r["song"]))
    return d


def run_algorithm_comparison(
    nrows: int | None = None,
    min_interactions: int = 8,
    test_ratio: float = 0.2,
    max_users: int = 120,
    max_samples: int = 200,
    negatives: int = 99,
    topk: int = 10,
    with_ncf: bool = True,
    include_hybrid: bool = True,
    hybrid_weights: dict | None = None,
    seed: int = 42,
    include_baselines: bool = True,
    use_persisted_models: bool = False,
):
    """
    运行多算法离线对比，返回指标字典与汇总表（供 Streamlit / 脚本共用）。

    Returns:
        summary: dict，含
          - df_main: 含 UserCF / ItemCF / SVD / 内容 / NCF / 融合（可选）
          - df_baseline: random / popular（若 include_baselines）
          - meta: 协议说明、有效样本数等
    """
    if nrows is None:
        nrows = CF_SAMPLE_SIZE

    random.seed(seed)
    np.random.seed(seed)
    global RNG
    RNG = np.random.default_rng(seed)

    df = load_interactions(nrows)
    all_songs = np.array(sorted(df["song"].unique()))

    train_df, test_pairs = build_train_test(
        df,
        min_interactions=min_interactions,
        test_ratio=test_ratio,
        max_test_users=max_users,
        max_test_samples=max_samples,
    )

    user_seen = user_train_songs(train_df)
    pop_scores = popularity_scores(train_df)

    if use_persisted_models:
        from src import recommend_utils as ru

        cf_payload = ru._get_cf_models()
        usercf = cf_payload["usercf"]
        itemcf = cf_payload["itemcf"]
        svd = cf_payload["svd"]
        artist_hasher, title_hasher, release_hasher, scaler, songs_meta = ru._get_feature_encoder()
    else:
        usercf, itemcf, svd, _ = fit_surprise_models(train_df, seed=seed)
        try:
            df_meta = pd.read_csv(
                DATA_FILE,
                nrows=nrows,
                usecols=["song", "artist_name", "title", "release", "year", "artist_familiarity", "artist_hotttnesss"],
            )
        except ValueError:
            df_meta = pd.read_csv(
                DATA_FILE,
                nrows=nrows,
                usecols=["song", "artist_name", "title", "year", "artist_familiarity", "artist_hotttnesss"],
            )
            df_meta["release"] = "unknown"
        artist_hasher, title_hasher, release_hasher, scaler, songs_meta = build_content_encoder(df_meta)

    ncf_model = None
    ncf_num_users = ncf_num_songs = None
    ncf_user2idx = {}
    ncf_song2idx = {}
    ncf_error = None
    if with_ncf:
        try:
            import torch
            if use_persisted_models:
                from src import recommend_utils as ru

                ncf_payload = ru._get_ncf_runtime()
                ncf_model = ncf_payload.get("model")
                ncf_num_users = int(ncf_payload.get("num_users", 0))
                ncf_num_songs = int(ncf_payload.get("num_songs", 0))
                ncf_user2idx = _normalize_idx_map(ncf_payload.get("user2idx", {}) or {})
                ncf_song2idx = _normalize_idx_map(ncf_payload.get("song2idx", {}) or {})
            else:
                # 同规模公平对比：NCF 现场训练（仅内存，不覆盖磁盘全局模型）
                ncf_model, ncf_num_users, ncf_num_songs, ncf_user2idx, ncf_song2idx = train_ncf_in_memory(
                    train_df=train_df,
                    seed=seed,
                )
            if ncf_model is not None:
                ncf_model.eval()
        except Exception as e:
            ncf_error = str(e)
            ncf_model = None

    _m0 = lambda: {"hr": [], "ndcg": [], "p": [], "recall": [], "mrr": [], "f1": []}
    metrics = {
        "random": _m0(),
        "popular": _m0(),
        "usercf": _m0(),
        "itemcf": _m0(),
        "svd": _m0(),
        "content": _m0(),
    }
    if ncf_model is not None:
        metrics["ncf"] = _m0()
    if include_hybrid:
        metrics["hybrid"] = _m0()
        metrics["hybrid"]["auc"] = []

    hybrid_weights = _normalize_hybrid_weights_five(hybrid_weights)

    def _log_ranking(ranked, bucket: str, pos_item: int):
        m = metrics[bucket]
        m["hr"].append(_hit_at_k(ranked, pos_item, topk))
        m["ndcg"].append(_ndcg_at_k(ranked, pos_item, topk))
        m["p"].append(_precision_at_k(ranked, pos_item, topk))
        m["recall"].append(_recall_at_k(ranked, pos_item, topk))
        m["mrr"].append(_mrr(ranked, pos_item))
        m["f1"].append(_f1_at_k(ranked, pos_item, topk))

    skipped = 0
    for user, pos in test_pairs:
        seen = user_seen[user]
        neg_pool = [s for s in all_songs if s not in seen and s != pos]
        if len(neg_pool) < negatives:
            skipped += 1
            continue
        negs = list(RNG.choice(neg_pool, size=negatives, replace=False))
        candidates = [pos] + [int(x) for x in negs]

        if include_baselines:
            shuffled = candidates.copy()
            random.shuffle(shuffled)
            _log_ranking(shuffled, "random", pos)

            scores_pop = {c: pop_scores.get(c, 0.0) for c in candidates}
            ranked_pop = sorted(candidates, key=lambda x: scores_pop[x], reverse=True)
            _log_ranking(ranked_pop, "popular", pos)

        sc_usercf, sc_itemcf, sc_svd = {}, {}, {}
        for name, algo in [("usercf", usercf), ("itemcf", itemcf), ("svd", svd)]:
            sc = {}
            for c in candidates:
                try:
                    p = algo.predict(user, c, clip=False)
                    sc[c] = float(p.est)
                except Exception:
                    sc[c] = 0.0
            ranked = sorted(candidates, key=lambda x: sc[x], reverse=True)
            _log_ranking(ranked, name, pos)
            if name == "usercf":
                sc_usercf = sc
            elif name == "itemcf":
                sc_itemcf = sc
            elif name == "svd":
                sc_svd = sc

        train_songs_list = list(seen)
        prof = user_content_profile(train_songs_list, artist_hasher, title_hasher, release_hasher, scaler, songs_meta)
        if prof is None:
            for key in ("hr", "ndcg", "p", "recall", "mrr", "f1"):
                metrics["content"][key].append(0.0)
            sc_c = {c: -1.0 for c in candidates}
        else:
            hist_artist = []
            hist_year = []
            if len(train_songs_list) > 0:
                hist_rows = songs_meta.reindex(train_songs_list)
                if "artist_name" in hist_rows.columns:
                    hist_artist = hist_rows["artist_name"].dropna().astype(str).tolist()
                if "year" in hist_rows.columns:
                    hist_year = pd.to_numeric(hist_rows["year"], errors="coerce").dropna().tolist()
            top_artists = set(pd.Series(hist_artist).value_counts().head(3).index.tolist()) if hist_artist else set()
            target_year = float(np.median(hist_year)) if hist_year else None

            sc_c = {}
            for c in candidates:
                v = song_vector(c, artist_hasher, title_hasher, release_hasher, scaler, songs_meta)
                if v is None:
                    sc_c[c] = -1.0
                    continue
                sim = cosine_sim(prof, v)
                bonus = 0.0
                if c in songs_meta.index:
                    row = songs_meta.loc[c]
                    artist = str(row.get("artist_name", "unknown")) if pd.notna(row.get("artist_name", None)) else "unknown"
                    if artist in top_artists:
                        bonus += 0.08
                    if target_year is not None:
                        year_val = row.get("year", np.nan)
                        if pd.notna(year_val):
                            year_aff = float(np.clip(1.0 - abs(float(year_val) - target_year) / 20.0, 0.0, 1.0))
                            bonus += 0.05 * year_aff
                sc_c[c] = float(sim + bonus)
            ranked_c = sorted(candidates, key=lambda x: sc_c[x], reverse=True)
            _log_ranking(ranked_c, "content", pos)

        if ncf_model is not None:
            import torch

            sc_n = {c: 0.0 for c in candidates}
            # 兼容两种 NCF 索引空间：
            # 1) 旧格式：原始 ID 直接作为 embedding 索引；
            # 2) 新格式：通过 user2idx/song2idx 做密集编码。
            if ncf_user2idx:
                user_enc = ncf_user2idx.get(int(user))
            else:
                user_enc = int(user) if int(user) < int(ncf_num_users) else None
            if user_enc is not None:
                valid_pairs = []
                for c in candidates:
                    if ncf_song2idx:
                        song_enc = ncf_song2idx.get(int(c))
                    else:
                        song_enc = int(c) if int(c) < int(ncf_num_songs) else None
                    if song_enc is not None:
                        valid_pairs.append((int(c), int(song_enc)))
                if valid_pairs:
                    raw_sids = [p[0] for p in valid_pairs]
                    enc_sids = [p[1] for p in valid_pairs]
                    ncf_device = next(ncf_model.parameters()).device
                    with torch.no_grad():
                        u_t = torch.tensor(
                            [int(user_enc)] * len(enc_sids),
                            dtype=torch.long,
                            device=ncf_device,
                        )
                        s_t = torch.tensor(enc_sids, dtype=torch.long, device=ncf_device)
                        out = ncf_model(u_t, s_t).detach().cpu().numpy()
                    for i, raw_sid in enumerate(raw_sids):
                        sc_n[raw_sid] = float(out[i])
            ranked_n = sorted(candidates, key=lambda x: sc_n[x], reverse=True)
            _log_ranking(ranked_n, "ncf", pos)
        else:
            sc_n = {c: 0.0 for c in candidates}

        # 融合评测：五路分数在候选集上分别做鲁棒校准后与线上一致加权；并计算候选级 AUC 供融合行专用展示
        if include_hybrid:
            order = list(candidates)
            chans = ("usercf", "itemcf", "svd", "ncf", "content")
            src = {
                "usercf": sc_usercf,
                "itemcf": sc_itemcf,
                "svd": sc_svd,
                "ncf": sc_n,
                "content": sc_c,
            }
            cal = {}
            for ch in chans:
                raw = [float(src[ch].get(c, 0.0)) for c in order]
                cal[ch] = dict(zip(order, _calibrate_scores_list(raw)))

            w = dict(hybrid_weights)
            if ncf_model is None:
                w["ncf"] = 0.0
            w = _normalize_hybrid_weights_five(w)

            sc_h = {}
            for c in candidates:
                num = 0.0
                den = 0.0
                for ch in chans:
                    wt = float(w.get(ch, 0.0))
                    if wt <= 0:
                        continue
                    num += wt * cal[ch].get(c, 0.0)
                    den += wt
                sc_h[c] = num / den if den > 1e-8 else 0.0

            ranked_h = sorted(candidates, key=lambda x: sc_h[x], reverse=True)
            _log_ranking(ranked_h, "hybrid", pos)

            labels_auc = [1] + [0] * len(negs)
            scores_auc = [sc_h[pos]] + [sc_h[int(x)] for x in negs]
            try:
                auc_v = float(roc_auc_score(labels_auc, scores_auc))
            except ValueError:
                auc_v = 0.5
            metrics["hybrid"]["auc"].append(auc_v)

    def _rows_from_metrics(keys, labels):
        rows = []
        for k in keys:
            if k not in metrics or not metrics[k]["hr"]:
                continue
            hr = float(np.mean(metrics[k]["hr"]))
            nd = float(np.mean(metrics[k]["ndcg"]))
            pr = float(np.mean(metrics[k]["p"]))
            rc = float(np.mean(metrics[k]["recall"]))
            mr = float(np.mean(metrics[k]["mrr"]))
            f1 = float(np.mean(metrics[k]["f1"]))
            row = {
                "算法": labels.get(k, k),
                f"HR@{topk}": hr,
                f"Recall@{topk}": rc,
                f"P@{topk}": pr,
                f"NDCG@{topk}": nd,
                "MRR": mr,
                f"F1@{topk}": f1,
                "HR(%)": round(hr * 100, 2),
                "Recall(%)": round(rc * 100, 2),
                "P(%)": round(pr * 100, 4),
                "NDCG(%)": round(nd * 100, 2),
                "MRR(%)": round(mr * 100, 2),
                "F1(%)": round(f1 * 100, 2),
            }
            if k == "hybrid" and metrics[k].get("auc"):
                row["候选AUC"] = round(float(np.mean(metrics[k]["auc"])), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    five_keys = ["usercf", "itemcf", "svd", "content", "ncf"]
    if include_hybrid:
        five_keys.append("hybrid")
    five_labels = {
        "usercf": "UserCF（用户协同）",
        "itemcf": "ItemCF（物品协同）",
        "svd": "SVD（矩阵分解）",
        "content": "内容推荐",
        "ncf": "NCF（神经协同过滤）",
        "hybrid": "融合推荐",
    }
    df_main = _rows_from_metrics(five_keys, five_labels)
    if with_ncf and ncf_model is None:
        df_main = pd.concat(
            [
                df_main,
                pd.DataFrame(
                    [
                        {
                            "算法": "NCF（神经协同过滤）",
                            f"HR@{topk}": np.nan,
                            f"Recall@{topk}": np.nan,
                            f"P@{topk}": np.nan,
                            f"NDCG@{topk}": np.nan,
                            "MRR": np.nan,
                            f"F1@{topk}": np.nan,
                            "HR(%)": np.nan,
                            "Recall(%)": np.nan,
                            "P(%)": np.nan,
                            "NDCG(%)": np.nan,
                            "MRR(%)": np.nan,
                            "F1(%)": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    df_baseline = pd.DataFrame()
    if include_baselines:
        df_baseline = _rows_from_metrics(
            ["random", "popular"],
            {"random": "随机基线", "popular": "热门基线"},
        )

    effective_n = len(test_pairs) - skipped
    protocol = (
        f"隐式反馈评测：每个测试样本候选集大小为 {negatives + 1}（1 正例 + {negatives} 负例），"
        f"展示指标 HR@K、NDCG@K、MRR、F1@K、准确率(P@K)，K={topk}；"
        f"融合行另含候选AUC（基于融合分在正负样本上的 ROC AUC）。"
        f"训练交互约 {len(train_df)} 条，有效测试样本 {effective_n}，跳过 {skipped}。"
    )

    return {
        "df_main": df_main,
        "df_baseline": df_baseline,
        "meta": {
            "protocol": protocol,
            "effective_samples": effective_n,
            "skipped": skipped,
            "input_nrows": int(nrows),
            "train_rows": len(train_df),
            "candidate_size": int(negatives + 1),
            "negatives": int(negatives),
            "include_baselines": bool(include_baselines),
            "use_persisted_models": bool(use_persisted_models),
            "train_unique_users": int(train_df["user"].nunique()),
            "train_unique_songs": int(train_df["song"].nunique()),
            "topk": topk,
            "ncf_loaded": ncf_model is not None,
            "ncf_error": ncf_error,
            "hybrid_weights": hybrid_weights if include_hybrid else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="推荐算法离线对比")
    parser.add_argument("--nrows", type=int, default=CF_SAMPLE_SIZE, help="读取交互行数")
    parser.add_argument("--min-interactions", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-users", type=int, default=200, help="参与划分的最大用户数")
    parser.add_argument("--max-samples", type=int, default=400, help="最多评估的测试样本数")
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("-k", "--k", type=int, default=10, dest="topk")
    parser.add_argument("--with-ncf", action="store_true", help="加载预训练 NCF（需 model/ncf_model.pth）")
    args = parser.parse_args()

    out = run_algorithm_comparison(
        nrows=args.nrows,
        min_interactions=args.min_interactions,
        test_ratio=args.test_ratio,
        max_users=args.max_users,
        max_samples=args.max_samples,
        negatives=args.negatives,
        topk=args.topk,
        with_ncf=args.with_ncf,
        seed=42,
        include_baselines=True,
    )
    meta = out["meta"]
    print(f"\n读取数据: {DATA_FILE}，{meta['protocol']}")

    print("\n" + "=" * 60)
    print("五类主算法对比（UserCF / ItemCF / SVD / 内容 / NCF）")
    print("=" * 60)
    print(out["df_main"].to_string(index=False))
    if not out["df_baseline"].empty:
        print("\n参考基线")
        print(out["df_baseline"].to_string(index=False))
    print("-" * 60)
    print("展示指标：HR@K、NDCG@K、MRR、F1@K、准确率(P@K)；融合行另含候选AUC。")
    if meta.get("ncf_error"):
        print(f"NCF 未加载: {meta['ncf_error']}")


if __name__ == "__main__":
    main()
