"""
训练并持久化模型 + 抽样10个训练用户验证。

用法：
  python train_and_validate_models.py
  python train_and_validate_models.py --users 10 --k 10 --negatives 99
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (  # noqa: E402
    DATA_FILE,
    CF_SAMPLE_SIZE,
    CONTENT_SAMPLE_SIZE,
    NCF_SAMPLE_SIZE,
    CF_USERCF_MODEL_PATH,
    CF_ITEMCF_MODEL_PATH,
    CF_SVD_MODEL_PATH,
    CF_META_PATH,
    CONTENT_MODEL_BUNDLE_PATH,
    NCF_MODEL_PATH,
)
from src import recommend_utils as ru  # noqa: E402


def _hit_at_k(ranked_items: list[int], positive: int, k: int) -> float:
    return 1.0 if positive in ranked_items[:k] else 0.0


def _precision_at_k(ranked_items: list[int], positive: int, k: int) -> float:
    return (1.0 / k) if positive in ranked_items[:k] else 0.0


def _ndcg_at_k(ranked_items: list[int], positive: int, k: int) -> float:
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == positive:
            return float((1.0 / np.log2(rank + 1)) / (1.0 / np.log2(2)))
    return 0.0


def _f1_at_k(ranked_items: list[int], positive: int, k: int) -> float:
    if positive not in ranked_items[:k]:
        return 0.0
    p = 1.0 / k
    r = 1.0
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def _metric_summary(records):
    if not records:
        return {"HR(%)": 0.0, "准确率(%)": 0.0, "NDCG(%)": 0.0, "F1(%)": 0.0}
    arr = np.asarray(records, dtype=np.float64)
    return {
        "HR(%)": round(float(np.mean(arr[:, 0])) * 100, 2),
        "准确率(%)": round(float(np.mean(arr[:, 1])) * 100, 4),
        "NDCG(%)": round(float(np.mean(arr[:, 2])) * 100, 2),
        "F1(%)": round(float(np.mean(arr[:, 3])) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="训练并持久化模型，抽样10用户验证")
    parser.add_argument("--users", type=int, default=10, help="验证用户数")
    parser.add_argument("--k", type=int, default=10, help="Top-K")
    parser.add_argument("--negatives", type=int, default=99, help="每用户负样本数量")
    args = parser.parse_args()

    print("=" * 72)
    print("模型训练与验证")
    print("=" * 72)
    print(f"数据文件: {DATA_FILE}")
    print("当前配置训练数据量上限:")
    print(f"  - CF_SAMPLE_SIZE: {CF_SAMPLE_SIZE}")
    print(f"  - CONTENT_SAMPLE_SIZE: {CONTENT_SAMPLE_SIZE}")
    print(f"  - NCF_SAMPLE_SIZE: {NCF_SAMPLE_SIZE}")
    print("-" * 72)

    # 触发训练并持久化（CF/内容）
    cf_payload = ru._get_cf_models()
    ru._get_feature_encoder()
    print("模型持久化文件:")
    print(f"  - UserCF: {CF_USERCF_MODEL_PATH} ({'存在' if os.path.isfile(CF_USERCF_MODEL_PATH) else '未生成'})")
    print(f"  - ItemCF: {CF_ITEMCF_MODEL_PATH} ({'存在' if os.path.isfile(CF_ITEMCF_MODEL_PATH) else '未生成'})")
    print(f"  - SVD: {CF_SVD_MODEL_PATH} ({'存在' if os.path.isfile(CF_SVD_MODEL_PATH) else '未生成'})")
    print(f"  - CF Meta: {CF_META_PATH} ({'存在' if os.path.isfile(CF_META_PATH) else '未生成'})")
    print(f"  - Content: {CONTENT_MODEL_BUNDLE_PATH} ({'存在' if os.path.isfile(CONTENT_MODEL_BUNDLE_PATH) else '未生成'})")
    print(f"  - NCF: {NCF_MODEL_PATH} ({'存在' if os.path.isfile(NCF_MODEL_PATH) else '未生成'})")
    print("-" * 72)

    df_cf = cf_payload["df_cf"]
    usercf = cf_payload["usercf"]
    itemcf = cf_payload["itemcf"]
    svd = cf_payload["svd"]

    user_groups = df_cf.groupby("user")["song"].apply(list)
    eligible_users = [int(u) for u, songs in user_groups.items() if len(set(songs)) >= 6]
    if len(eligible_users) == 0:
        print("可验证用户不足（至少需要 6 条历史）。")
        return

    rng = np.random.default_rng(42)
    rng.shuffle(eligible_users)
    users = eligible_users[: max(1, args.users)]
    all_songs = np.array(sorted(df_cf["song"].unique()))

    artist_hasher, scaler, songs_meta = ru._get_feature_encoder()
    ncf_payload = ru._get_ncf_runtime()
    ncf_model = ncf_payload.get("model")
    ncf_num_songs = int(ncf_payload.get("num_songs", 0))

    rec_cf, rec_item, rec_svd, rec_content, rec_ncf = [], [], [], [], []

    for user in users:
        hist = list(dict.fromkeys([int(x) for x in user_groups[user]]))
        pos = hist[-1]
        context = hist[:-1]
        if len(context) < 3:
            continue
        seen = set(hist)
        neg_pool = [s for s in all_songs if s not in seen]
        if len(neg_pool) < args.negatives:
            continue
        negs = list(rng.choice(neg_pool, size=args.negatives, replace=False))
        candidates = [pos] + negs

        # CF / ItemCF / SVD
        for algo, bucket in ((usercf, rec_cf), (itemcf, rec_item), (svd, rec_svd)):
            sc = {c: float(algo.predict(user, c, clip=False).est) for c in candidates}
            ranked = sorted(candidates, key=lambda x: sc[x], reverse=True)
            bucket.append((
                _hit_at_k(ranked, pos, args.k),
                _precision_at_k(ranked, pos, args.k),
                _ndcg_at_k(ranked, pos, args.k),
                _f1_at_k(ranked, pos, args.k),
            ))

        # Content
        prof_feats = []
        for sid in context:
            v = ru._get_song_feature(sid, artist_hasher, scaler, songs_meta)
            if v is not None:
                prof_feats.append(v)
        if prof_feats:
            prof = np.mean(prof_feats, axis=0)
            sc_c = {}
            for c in candidates:
                v = ru._get_song_feature(c, artist_hasher, scaler, songs_meta)
                if v is None:
                    sc_c[c] = -1.0
                else:
                    sc_c[c] = float(np.dot(prof, v) / (np.linalg.norm(prof) * np.linalg.norm(v) + 1e-8))
            ranked_c = sorted(candidates, key=lambda x: sc_c[x], reverse=True)
            rec_content.append((
                _hit_at_k(ranked_c, pos, args.k),
                _precision_at_k(ranked_c, pos, args.k),
                _ndcg_at_k(ranked_c, pos, args.k),
                _f1_at_k(ranked_c, pos, args.k),
            ))

        # NCF（若已存在）
        if ncf_model is not None and user < int(ncf_payload.get("num_users", 0)) and all(c < ncf_num_songs for c in candidates):
            import torch

            with torch.no_grad():
                u_t = torch.tensor([user] * len(candidates), dtype=torch.long)
                s_t = torch.tensor(candidates, dtype=torch.long)
                out = ncf_model(u_t, s_t).numpy()
            sc_n = {candidates[i]: float(out[i]) for i in range(len(candidates))}
            ranked_n = sorted(candidates, key=lambda x: sc_n[x], reverse=True)
            rec_ncf.append((
                _hit_at_k(ranked_n, pos, args.k),
                _precision_at_k(ranked_n, pos, args.k),
                _ndcg_at_k(ranked_n, pos, args.k),
                _f1_at_k(ranked_n, pos, args.k),
            ))

    print(f"验证用户数: {len(users)}")
    print("验证结果（抽样训练用户，统一候选集）:")
    print(f"  - UserCF: {_metric_summary(rec_cf)}")
    print(f"  - ItemCF: {_metric_summary(rec_item)}")
    print(f"  - SVD: {_metric_summary(rec_svd)}")
    print(f"  - Content: {_metric_summary(rec_content)}")
    if ncf_model is not None:
        print(f"  - NCF: {_metric_summary(rec_ncf)}")
    else:
        print("  - NCF: 模型文件不存在或未加载，跳过。")
    print("=" * 72)


if __name__ == "__main__":
    main()

