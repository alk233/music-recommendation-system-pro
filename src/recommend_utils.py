# 推荐算法工具模块
import pandas as pd
import numpy as np
import torch
import streamlit as st
import pickle
import json
from surprise import Dataset, Reader, KNNBasic, SVD as SurpriseSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from surprise.model_selection import train_test_split
import sys
import os
from typing import Optional

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    DATA_FILE,
    NCF_MODEL_PATH,
    NCF_PERSONALIZED_MODEL_PATH,
    CF_USERCF_MODEL_PATH,
    CF_ITEMCF_MODEL_PATH,
    CF_SVD_MODEL_PATH,
    CF_META_PATH,
    CONTENT_MODEL_BUNDLE_PATH,
    HISTORY_FILE,
    USERS_FILE,
    CF_SAMPLE_SIZE,
    CF_KNN_K,
    CF_MIN_EST_SCORE,
    NCF_SAMPLE_SIZE,
    CONTENT_SAMPLE_SIZE,
    STREAMLIT_DATA_NROWS,
)
from src.deep_learning_recommend import (
    NCF,
    _sample_bpr_triplets,
    _build_user_sampling_tables,
    _build_song_sampling_probs,
)

# 个性化 NCF 训练时，从全局交互中随机抽样的上限
PERSONALIZED_NCF_GLOBAL_SAMPLE = 50000
_DATA_TOTAL_ROWS_CACHE = None


def _load_torch_state_dict(path: str, map_location: str = "cpu"):
    # 优先完整加载（保留 checkpoint 元数据如 user2idx/song2idx）；
    # 若本地 torch 受限再回退到 weights_only=True。
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _normalize_idx_map(m):
    """将映射表统一规范为 int->int，兼容 checkpoint 中可能出现的字符串键。"""
    if not isinstance(m, dict):
        return {}
    out = {}
    for k, v in m.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


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


def _safe_pickle_load(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _safe_pickle_dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_registered_history_interactions():
    """读取 users.json + user_history.json，返回 (user, song, play_count=1.0) 交互表。"""
    if not (os.path.isfile(USERS_FILE) and os.path.isfile(HISTORY_FILE)):
        return pd.DataFrame(columns=["user", "song", "play_count"])
    try:
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
    return pd.DataFrame(pairs, columns=["user", "song", "play_count"]).drop_duplicates(["user", "song"])


def _lookup_song_meta_from_disk(song_ids) -> dict:
    """对给定 song_id 在主表中分块查找 title/artist_name（用于仅出现在注册用户历史中的歌曲）。"""
    need = set()
    for x in song_ids:
        try:
            if pd.isna(x):
                continue
            need.add(int(x))
        except (TypeError, ValueError):
            continue
    if not need:
        return {}
    out = {}
    usecols = ["song", "title", "artist_name"]
    try:
        reader = pd.read_csv(DATA_FILE, usecols=usecols, chunksize=200_000)
    except ValueError:
        return out
    for chunk in reader:
        sub = chunk[chunk["song"].isin(need)]
        if sub.empty:
            continue
        for _, row in sub.drop_duplicates("song").iterrows():
            sid = int(row["song"])
            if sid in out:
                continue
            out[sid] = {"title": row.get("title"), "artist_name": row.get("artist_name")}
        if need <= set(out.keys()):
            break
    return out


def _append_registered_history(df: pd.DataFrame) -> pd.DataFrame:
    """将注册用户历史临时并入训练交互（不修改主 CSV）。"""
    ext = _load_registered_history_interactions()
    if ext.empty:
        return df
    base_cols = [c for c in ["user", "song", "play_count"] if c in df.columns]
    if len(base_cols) < 3:
        return df
    meta_cols = [c for c in df.columns if c not in base_cols]
    merged = pd.concat([df[base_cols].copy(), ext], ignore_index=True)
    merged = merged.groupby(["user", "song"], as_index=False)["play_count"].max()
    if meta_cols:
        song_meta = df.drop_duplicates(subset=["song"], keep="first")[["song"] + meta_cols]
        merged = merged.merge(song_meta, on="song", how="left")
        if "title" in merged.columns:
            miss_ids = [int(x) for x in merged.loc[merged["title"].isna(), "song"].dropna().unique().tolist()]
            disk = _lookup_song_meta_from_disk(miss_ids)
            for sid, meta in disk.items():
                m = merged["song"] == sid
                if not m.any():
                    continue
                t, a = meta.get("title"), meta.get("artist_name")
                if pd.isna(merged.loc[m, "title"]).all() and t is not None and pd.notna(t):
                    merged.loc[m, "title"] = t
                if "artist_name" in merged.columns and pd.isna(merged.loc[m, "artist_name"]).all() and a is not None and pd.notna(a):
                    merged.loc[m, "artist_name"] = a
    else:
        for col in df.columns:
            if col not in merged.columns:
                merged[col] = pd.NA
    for col in df.columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[df.columns]


def _history_song_popularity() -> dict:
    """所有注册用户历史歌曲的频次（用于内容推荐的轻量先验）。"""
    ext = _load_registered_history_interactions()
    if ext.empty:
        return {}
    return ext.groupby("song").size().astype(float).to_dict()


def _count_data_rows_once() -> int:
    """统计 DATA_FILE 数据行数（不含表头），进程内缓存一次。"""
    global _DATA_TOTAL_ROWS_CACHE
    if _DATA_TOTAL_ROWS_CACHE is not None:
        return _DATA_TOTAL_ROWS_CACHE
    cnt = 0
    with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)  # 跳过表头
        for _ in f:
            cnt += 1
    _DATA_TOTAL_ROWS_CACHE = cnt
    return cnt


def _read_random_ncf_window(sample_size: int, seed: int) -> pd.DataFrame:
    """
    从主表中随机读取一段连续窗口，避免“先读大表再采样”的额外开销。
    返回 user/song/play_count 三列；若主表缺少 play_count 列则回填为 1.0。
    """
    total_rows = _count_data_rows_once()
    if total_rows <= 0:
        return pd.DataFrame(columns=["user", "song", "play_count"])
    take = int(min(max(1, sample_size), total_rows))
    if total_rows <= take:
        try:
            out = pd.read_csv(
                DATA_FILE,
                usecols=["user", "song", "play_count"],
                dtype={"user": "int32", "song": "int32", "play_count": "float32"},
            )
        except ValueError:
            out = pd.read_csv(
                DATA_FILE,
                usecols=["user", "song"],
                dtype={"user": "int32", "song": "int32"},
            )
            out["play_count"] = 1.0
        out["play_count"] = pd.to_numeric(out["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
        return out
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, total_rows - take + 1))
    # skiprows 需包含表头后的行号（从 1 开始）
    try:
        out = pd.read_csv(
            DATA_FILE,
            usecols=["user", "song", "play_count"],
            dtype={"user": "int32", "song": "int32", "play_count": "float32"},
            skiprows=range(1, start + 1),
            nrows=take,
        )
    except ValueError:
        out = pd.read_csv(
            DATA_FILE,
            usecols=["user", "song"],
            dtype={"user": "int32", "song": "int32"},
            skiprows=range(1, start + 1),
            nrows=take,
        )
        out["play_count"] = 1.0
    out["play_count"] = pd.to_numeric(out["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    return out

# Surprise 余弦相似度在「全零评分向量」上会除零；QuantileTransformer 可产生 0，需抬高下界
CF_RATING_EPS = 1e-6


def _clip_cf_play_count(df, col="play_count"):
    out = df.copy()
    out[col] = np.maximum(out[col].astype(np.float64), CF_RATING_EPS)
    return out


def _build_cf_song_info_dict(df_cf: pd.DataFrame) -> dict:
    """由协同过滤数据框构建 song_id -> 展示文案，避免 pandas NA 被格式化成字面量 <NA>。"""
    if df_cf is None or df_cf.empty or "song" not in df_cf.columns:
        return {}
    cols = ["song", "title", "artist_name"]
    if not all(c in df_cf.columns for c in cols):
        return {}
    song_info = df_cf.drop_duplicates("song")[cols]

    def _one(row) -> str:
        a = row["artist_name"]
        t = row["title"]
        if a is None or (isinstance(a, str) and not a.strip()) or pd.isna(a):
            a = "未知"
        else:
            a = str(a)
        if t is None or (isinstance(t, str) and not t.strip()) or pd.isna(t):
            t = "未知"
        else:
            t = str(t)
        return f"{a} - {t}"

    return song_info.set_index("song").apply(_one, axis=1).to_dict()


def _safe_reader_bounds(s):
    lo = float(np.min(s))
    hi = float(np.max(s))
    if hi <= lo:
        hi = lo + CF_RATING_EPS
    return lo, hi


# 原始标签 z 挤在约 [0,1] 时，KNN 加权平均易退化为常数；训练前线性映射到更宽区间提升分差
CF_SURPRISE_RATING_LO = 1.0
CF_SURPRISE_RATING_HI = 10.0
# 针对当前数据稀疏度的 SVD 稳定参数（不依赖时间戳）
CF_SVD_FACTORS = 160
CF_SVD_EPOCHS = 36
CF_SVD_LR_ALL = 0.0025
CF_SVD_REG_ALL = 0.03
CF_SVD_RATING_TRANSFORM_VERSION = 3
CF_KNN_MIN_SUPPORT = 2

# 历史兼容参数：曾用于控制实体截断。当前口径已关闭实体截断（保留常量仅为兼容旧代码）。
CF_MAX_UNIQUE_USERS = 9000
CF_MAX_UNIQUE_SONGS = 9000
CF_USERCF_SAMPLE_SIZE = CF_SAMPLE_SIZE
CF_ITEMCF_SAMPLE_SIZE = CF_SAMPLE_SIZE
CF_SVD_SAMPLE_SIZE = CF_SAMPLE_SIZE


def _sync_cf_sample_sizes_from_persisted_meta(cached_meta: Optional[dict]) -> None:
    """若磁盘上已有 CF 元数据与模型文件，将内存中的读取条数与元数据对齐，避免进程重启后配置默认值触发整包重训。"""
    global CF_SAMPLE_SIZE, CF_USERCF_SAMPLE_SIZE, CF_ITEMCF_SAMPLE_SIZE, CF_SVD_SAMPLE_SIZE
    if not isinstance(cached_meta, dict):
        return
    if not (
        os.path.isfile(CF_USERCF_MODEL_PATH)
        and os.path.isfile(CF_ITEMCF_MODEL_PATH)
        and os.path.isfile(CF_SVD_MODEL_PATH)
    ):
        return
    try:
        u = int(cached_meta.get("cf_usercf_sample_size", CF_USERCF_SAMPLE_SIZE))
        i = int(cached_meta.get("cf_itemcf_sample_size", CF_ITEMCF_SAMPLE_SIZE))
        s = int(cached_meta.get("cf_svd_sample_size", CF_SVD_SAMPLE_SIZE))
        CF_USERCF_SAMPLE_SIZE = max(1000, u)
        CF_ITEMCF_SAMPLE_SIZE = max(1000, i)
        CF_SVD_SAMPLE_SIZE = max(1000, s)
        CF_SAMPLE_SIZE = max(CF_USERCF_SAMPLE_SIZE, CF_ITEMCF_SAMPLE_SIZE, CF_SVD_SAMPLE_SIZE)
    except Exception:
        pass


def _limit_cf_entities(df: pd.DataFrame) -> pd.DataFrame:
    """限制训练集中的唯一 user/song 数量（当前仅用于 ItemCF 训练）。"""
    out = df
    try:
        n_users = int(out["user"].nunique())
        n_songs = int(out["song"].nunique())
    except Exception:
        return out

    if n_songs > CF_MAX_UNIQUE_SONGS:
        top_songs = (
            out.groupby("song")["play_count"].sum().sort_values(ascending=False).head(CF_MAX_UNIQUE_SONGS).index
        )
        out = out[out["song"].isin(top_songs)].copy()

    if n_users > CF_MAX_UNIQUE_USERS:
        top_users = (
            out.groupby("user")["play_count"].sum().sort_values(ascending=False).head(CF_MAX_UNIQUE_USERS).index
        )
        out = out[out["user"].isin(top_users)].copy()
    return out


def _stretch_cf_training_ratings(arr):
    """KNN 训练分数映射：保持线性拉伸，利于邻域法解释性。"""
    v = np.asarray(arr, dtype=np.float64)
    lo, hi = np.min(v), np.max(v)
    if hi <= lo:
        return np.full_like(v, 5.5)
    return CF_SURPRISE_RATING_LO + (CF_SURPRISE_RATING_HI - CF_SURPRISE_RATING_LO) * (v - lo) / (hi - lo)


def _stretch_svd_training_ratings(arr):
    """
    SVD 训练分数映射：与 KNN 统一为线性拉伸。
    实测该数据集在统一映射下更稳，避免过度压缩导致可分性下降。
    """
    return _stretch_cf_training_ratings(arr)


def _display_float(x) -> str:
    """列表展示用：不截断到 1、不固定小数位，按浮点实际值输出（便于区分）。"""
    v = float(np.asarray(x, dtype=np.float64).item())
    if not np.isfinite(v):
        return "—"
    return repr(v)


def _smooth_scale_scores(values, out_lo=0.0, out_hi=100.0):
    """
    将任意模型原始分平滑映射到统一展示刻度（默认 0~100）。
    仅用于展示，不参与排序与评测。
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [50.0 for _ in values]
    if hi <= lo:
        return [50.0 for _ in values]
    t = (arr - lo) / (hi - lo)
    # sigmoid 平滑，避免极值挤在 0/100
    t = 1.0 / (1.0 + np.exp(-6.0 * (t - 0.5)))
    out = out_lo + (out_hi - out_lo) * t
    return [float(x) for x in out]


def _calibrate_scores(values):
    """
    统一校准到 0~1（用于跨模型融合，不用于排序评测）。
    使用鲁棒 z 分数（median/IQR）+ sigmoid，减少极端值和量纲差异影响。
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
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


# 多维度榜单推荐（冷启动）
@st.cache_data
def _get_cold_start_data():
    # 缓存歌曲统计数据（包含多个维度）；限制行数避免全表读取拖慢首屏
    cols = [
        "song",
        "artist_name",
        "title",
        "year",
        "artist_hotttnesss",
        "artist_familiarity",
        "play_count",
    ]
    df = pd.read_csv(DATA_FILE, usecols=cols, nrows=STREAMLIT_DATA_NROWS)
    song_stats = (
        df.groupby(
            ["song", "artist_name", "title", "year", "artist_hotttnesss", "artist_familiarity"]
        )["play_count"]
        .sum()
        .reset_index()
    )
    return song_stats

def popularity_cold_start(topk=10, chart_type='popularity', diversity=True, seed=None):
    """
    多维度榜单推荐（冷启动）
    
    Args:
        topk: 推荐数量
        chart_type: 榜单类型
            - 'popularity': 热门度榜单（按样本内偏好强度汇总）
            - 'artist': 歌手榜单（按歌手热度）
        diversity: 是否使用多样性推荐（从Top歌曲中随机采样）
        seed: 随机种子，用于控制随机性
    """
    song_stats = _get_cold_start_data()
    
    # 根据榜单类型排序
    if chart_type == 'popularity':
        # 热门度榜单：按偏好强度（汇总）排序
        sorted_stats = song_stats.sort_values('play_count', ascending=False)
    elif chart_type == 'artist':
        # 歌手榜单：按歌手热度排序
        sorted_stats = song_stats.sort_values('artist_hotttnesss', ascending=False)
    else:
        # 默认热门度榜单
        sorted_stats = song_stats.sort_values('play_count', ascending=False)
    
    if diversity:
        # 多样性推荐：从Top歌曲中随机采样
        # 从Top 100歌曲中随机选择，保证质量和多样性
        top_n = min(100, len(sorted_stats))
        top_songs_pool = sorted_stats.head(top_n)
        
        # 使用随机种子确保可复现，但每次刷新可以不同
        if seed is None:
            import random
            seed = random.randint(0, 1000000)
        
        np.random.seed(seed)
        # 随机采样，不重复
        if len(top_songs_pool) >= topk:
            sampled_indices = np.random.choice(len(top_songs_pool), size=topk, replace=False)
            top_songs = top_songs_pool.iloc[sampled_indices]
        else:
            top_songs = top_songs_pool
    else:
        # 传统方式：直接返回Top-N
        top_songs = sorted_stats.head(topk)

    # 展示时始终按指标从高到低排序（多样性采样后原先顺序是乱的）
    if chart_type == 'popularity':
        top_songs = top_songs.sort_values('play_count', ascending=False)
    elif chart_type == 'artist':
        top_songs = top_songs.sort_values('artist_hotttnesss', ascending=False)
    
    # 格式化结果（列名仍为 play_count，值为 z=分位数(ln(2+r))，见界面「备注」）
    result = []
    for _, row in top_songs.iterrows():
        if chart_type == 'popularity':
            result.append(
                f"{row['artist_name']} - {row['title']} (song_id={row['song']})，"
                f"热度分 {row['play_count']:.2f}"
            )
        elif chart_type == 'artist':
            artist_hot = row['artist_hotttnesss'] if pd.notna(row['artist_hotttnesss']) else 0
            result.append(f"{row['artist_name']} - {row['title']} (song_id={row['song']})，歌手热度：{artist_hot:.4f}")
        else:
            result.append(
                f"{row['artist_name']} - {row['title']} (song_id={row['song']})，"
                f"热度分 {row['play_count']:.2f}"
            )
    
    return result

@st.cache_data
def _get_cf_data(nrows=None, apply_entity_limit: bool = False, _cf_data_cache_gen: int = 4):
    # 缓存协同过滤数据，避免每次刷新重复读取CSV（_cf_data_cache_gen 用于规则变更后失效旧缓存）
    if nrows is None:
        nrows = CF_SAMPLE_SIZE
    _ = _cf_data_cache_gen
    df = pd.read_csv(DATA_FILE, nrows=int(nrows))
    df = _append_registered_history(df)
    df = _clip_cf_play_count(df)
    return _limit_cf_entities(df) if apply_entity_limit else df


def retrain_cf_models(selected_models=None):
    """
    按模型粒度重训 CF 子模型，并仅覆盖被选中的模型文件。
    selected_models: 可迭代，元素取 usercf/itemcf/svd；为空时默认全量。
    返回与 _get_cf_models 一致的 payload。
    """
    all_models = ("usercf", "itemcf", "svd")
    if selected_models is None:
        selected = set(all_models)
    else:
        try:
            selected = {str(x).strip().lower() for x in selected_models}
        except Exception:
            selected = set(all_models)
        selected = {x for x in selected if x in all_models}
        if not selected:
            selected = set(all_models)

    # 先尝试加载现有模型（未选中的模型尽量复用，不覆盖）
    models = {
        "usercf": _safe_pickle_load(CF_USERCF_MODEL_PATH),
        "itemcf": _safe_pickle_load(CF_ITEMCF_MODEL_PATH),
        "svd": _safe_pickle_load(CF_SVD_MODEL_PATH),
    }

    # 若未选中模型缺失，则一并补训，保证接口返回完整
    for name in all_models:
        if models.get(name) is None:
            selected.add(name)

    df_usercf = _get_cf_data(CF_USERCF_SAMPLE_SIZE, apply_entity_limit=False) if "usercf" in selected else None
    # ItemCF 恢复实体截断：仅保留 Top 9000 用户/歌曲，避免相似度矩阵爆内存
    df_itemcf = _get_cf_data(CF_ITEMCF_SAMPLE_SIZE, apply_entity_limit=True) if "itemcf" in selected else None
    df_svd = _get_cf_data(CF_SVD_SAMPLE_SIZE, apply_entity_limit=False) if "svd" in selected else None
    df_cf = _get_cf_data(max(CF_USERCF_SAMPLE_SIZE, CF_ITEMCF_SAMPLE_SIZE, CF_SVD_SAMPLE_SIZE), apply_entity_limit=False)
    reader = Reader(rating_scale=(CF_SURPRISE_RATING_LO, CF_SURPRISE_RATING_HI))

    def _build_trainset(df_src, for_svd=False):
        df_s = df_src[["user", "song", "play_count"]].copy()
        if for_svd:
            df_s["play_count"] = _stretch_svd_training_ratings(df_s["play_count"].values)
        else:
            df_s["play_count"] = _stretch_cf_training_ratings(df_s["play_count"].values)
        data = Dataset.load_from_df(df_s, reader)
        return data.build_full_trainset()

    if "usercf" in selected:
        trainset_usercf = _build_trainset(df_usercf, for_svd=False)
        algo_usercf = KNNBasic(
            k=CF_KNN_K, sim_options={'name': 'cosine', 'user_based': True, 'min_support': CF_KNN_MIN_SUPPORT}
        )
        algo_usercf.fit(trainset_usercf)
        models["usercf"] = algo_usercf
        _safe_pickle_dump(CF_USERCF_MODEL_PATH, algo_usercf)

    if "itemcf" in selected:
        trainset_itemcf = _build_trainset(df_itemcf, for_svd=False)
        algo_itemcf = KNNBasic(
            k=CF_KNN_K, sim_options={'name': 'cosine', 'user_based': False, 'min_support': CF_KNN_MIN_SUPPORT}
        )
        algo_itemcf.fit(trainset_itemcf)
        models["itemcf"] = algo_itemcf
        _safe_pickle_dump(CF_ITEMCF_MODEL_PATH, algo_itemcf)

    if "svd" in selected:
        trainset_svd = _build_trainset(df_svd, for_svd=True)
        algo_svd = SurpriseSVD(
            n_factors=CF_SVD_FACTORS,
            n_epochs=CF_SVD_EPOCHS,
            lr_all=CF_SVD_LR_ALL,
            reg_all=CF_SVD_REG_ALL,
            random_state=42,
            biased=True,
        )
        algo_svd.fit(trainset_svd)
        models["svd"] = algo_svd
        _safe_pickle_dump(CF_SVD_MODEL_PATH, algo_svd)

    os.makedirs(os.path.dirname(CF_META_PATH), exist_ok=True)
    with open(CF_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cf_usercf_sample_size": int(CF_USERCF_SAMPLE_SIZE),
                "cf_itemcf_sample_size": int(CF_ITEMCF_SAMPLE_SIZE),
                "cf_svd_sample_size": int(CF_SVD_SAMPLE_SIZE),
                "cf_rating_lo": CF_SURPRISE_RATING_LO,
                "cf_rating_hi": CF_SURPRISE_RATING_HI,
                "cf_max_users": int(CF_MAX_UNIQUE_USERS),
                "cf_max_songs": int(CF_MAX_UNIQUE_SONGS),
                "cf_svd_factors": int(CF_SVD_FACTORS),
                "cf_svd_epochs": int(CF_SVD_EPOCHS),
                "cf_svd_lr_all": float(CF_SVD_LR_ALL),
                "cf_svd_reg_all": float(CF_SVD_REG_ALL),
                "cf_svd_rating_transform_version": int(CF_SVD_RATING_TRANSFORM_VERSION),
                "cf_knn_min_support": int(CF_KNN_MIN_SUPPORT),
                "cf_entity_limit_mode": "itemcf_only",
                "n_rows": int(len(df_cf)),
                "n_users": int(df_cf["user"].nunique()),
                "n_songs": int(df_cf["song"].nunique()),
                "last_retrained_models": sorted(list(selected)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    song_info_dict = _build_cf_song_info_dict(df_cf)
    return {
        "df_cf": df_cf,
        "usercf": models["usercf"],
        "itemcf": models["itemcf"],
        "svd": models["svd"],
        "song_info_dict": song_info_dict,
    }


@st.cache_resource
def _get_cf_models():
    # 懒加载并缓存协同过滤模型；UserCF/ItemCF/SVD 分文件保存，元数据放在 cf_meta.json
    cached_meta = {}
    if os.path.isfile(CF_META_PATH):
        try:
            with open(CF_META_PATH, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
        except Exception:
            cached_meta = {}
    _sync_cf_sample_sizes_from_persisted_meta(cached_meta)

    if (
        isinstance(cached_meta, dict)
        and int(cached_meta.get("cf_usercf_sample_size", -1)) == int(CF_USERCF_SAMPLE_SIZE)
        and int(cached_meta.get("cf_itemcf_sample_size", -1)) == int(CF_ITEMCF_SAMPLE_SIZE)
        and int(cached_meta.get("cf_svd_sample_size", -1)) == int(CF_SVD_SAMPLE_SIZE)
        and float(cached_meta.get("cf_rating_lo", -1)) == CF_SURPRISE_RATING_LO
        and float(cached_meta.get("cf_rating_hi", -1)) == CF_SURPRISE_RATING_HI
        and int(cached_meta.get("cf_max_users", -1)) == int(CF_MAX_UNIQUE_USERS)
        and int(cached_meta.get("cf_max_songs", -1)) == int(CF_MAX_UNIQUE_SONGS)
        and int(cached_meta.get("cf_svd_factors", -1)) == int(CF_SVD_FACTORS)
        and int(cached_meta.get("cf_svd_epochs", -1)) == int(CF_SVD_EPOCHS)
        and float(cached_meta.get("cf_svd_lr_all", -1)) == float(CF_SVD_LR_ALL)
        and float(cached_meta.get("cf_svd_reg_all", -1)) == float(CF_SVD_REG_ALL)
        and int(cached_meta.get("cf_svd_rating_transform_version", -1)) == int(CF_SVD_RATING_TRANSFORM_VERSION)
        and int(cached_meta.get("cf_knn_min_support", -1)) == int(CF_KNN_MIN_SUPPORT)
        and str(cached_meta.get("cf_entity_limit_mode", "legacy")) == "itemcf_only"
        and os.path.isfile(CF_USERCF_MODEL_PATH)
        and os.path.isfile(CF_ITEMCF_MODEL_PATH)
        and os.path.isfile(CF_SVD_MODEL_PATH)
    ):
        usercf = _safe_pickle_load(CF_USERCF_MODEL_PATH)
        itemcf = _safe_pickle_load(CF_ITEMCF_MODEL_PATH)
        svd = _safe_pickle_load(CF_SVD_MODEL_PATH)
        if usercf is not None and itemcf is not None and svd is not None:
            df_cf = _get_cf_data(max(CF_USERCF_SAMPLE_SIZE, CF_ITEMCF_SAMPLE_SIZE, CF_SVD_SAMPLE_SIZE), apply_entity_limit=False)
            song_info_dict = _build_cf_song_info_dict(df_cf)
            return {
                "df_cf": df_cf,
                "usercf": usercf,
                "itemcf": itemcf,
                "svd": svd,
                "song_info_dict": song_info_dict,
            }

    return retrain_cf_models(selected_models=("usercf", "itemcf", "svd"))


def _normalize_user_history(user_history):
    # 规范化用户历史，作为训练样本与缓存键
    if not user_history:
        return []
    normalized = []
    for song_id in user_history:
        try:
            normalized.append(int(song_id))
        except (TypeError, ValueError):
            continue
    # 去重并排序，避免历史顺序变化导致重复训练
    return sorted(set(normalized))


@st.cache_resource
def _get_personalized_cf_models(user_id, user_history_key, algo_name=None, nrows_override=None):
    # 将用户新增历史合并进训练集，按所选算法训练个性化CF模型
    algo = str(algo_name).strip().lower() if algo_name is not None else ""
    if algo not in {"usercf", "itemcf", "svd"}:
        algo = ""
    if algo == "usercf":
        nrows = int(CF_USERCF_SAMPLE_SIZE)
    elif algo == "itemcf":
        nrows = int(CF_ITEMCF_SAMPLE_SIZE)
    elif algo == "svd":
        nrows = int(CF_SVD_SAMPLE_SIZE)
    else:
        nrows = int(max(CF_USERCF_SAMPLE_SIZE, CF_ITEMCF_SAMPLE_SIZE, CF_SVD_SAMPLE_SIZE))
    if nrows_override is not None:
        try:
            nrows = max(1000, int(nrows_override))
        except Exception:
            pass
    df_cf = _get_cf_data(nrows=nrows, apply_entity_limit=False)
    user_id = int(user_id)
    history = [int(song_id) for song_id in user_history_key]

    df_train = df_cf[['user', 'song', 'play_count']].copy()
    if history:
        # 用户历史没有显式评分时，使用较高偏好强度作为正反馈
        user_rows = pd.DataFrame({
            'user': [user_id] * len(history),
            'song': history,
            'play_count': [1.0] * len(history)
        })
        df_train = pd.concat([df_train, user_rows], ignore_index=True)
        # 去重后保留最大偏好值，避免重复交互稀释权重
        df_train = df_train.groupby(['user', 'song'], as_index=False)['play_count'].max()

    df_train = _clip_cf_play_count(df_train)
    df_z = df_train[["user", "song", "play_count"]].copy()
    reader = Reader(rating_scale=(CF_SURPRISE_RATING_LO, CF_SURPRISE_RATING_HI))
    df_usercf_knn = df_z.copy()
    df_usercf_knn["play_count"] = _stretch_cf_training_ratings(df_usercf_knn["play_count"].values)
    data_usercf_knn = Dataset.load_from_df(df_usercf_knn, reader)
    trainset_usercf_knn = data_usercf_knn.build_full_trainset()
    # 个性化 ItemCF 同样使用实体截断，保持与全局 ItemCF 一致
    df_itemcf_knn = _limit_cf_entities(df_z.copy())
    df_itemcf_knn["play_count"] = _stretch_cf_training_ratings(df_itemcf_knn["play_count"].values)
    data_itemcf_knn = Dataset.load_from_df(df_itemcf_knn, reader)
    trainset_itemcf_knn = data_itemcf_knn.build_full_trainset()
    df_svd = df_z.copy()
    df_svd["play_count"] = _stretch_svd_training_ratings(df_svd["play_count"].values)
    data_svd = Dataset.load_from_df(df_svd, reader)
    trainset_svd = data_svd.build_full_trainset()

    algo_usercf = None
    algo_itemcf = None
    algo_svd = None
    if algo in ("", "usercf"):
        algo_usercf = KNNBasic(
            k=CF_KNN_K, sim_options={'name': 'cosine', 'user_based': True, 'min_support': CF_KNN_MIN_SUPPORT}
        )
        algo_usercf.fit(trainset_usercf_knn)

    if algo in ("", "itemcf"):
        algo_itemcf = KNNBasic(
            k=CF_KNN_K, sim_options={'name': 'cosine', 'user_based': False, 'min_support': CF_KNN_MIN_SUPPORT}
        )
        algo_itemcf.fit(trainset_itemcf_knn)

    if algo in ("", "svd"):
        algo_svd = SurpriseSVD(
            n_factors=CF_SVD_FACTORS,
            n_epochs=CF_SVD_EPOCHS,
            lr_all=CF_SVD_LR_ALL,
            reg_all=CF_SVD_REG_ALL,
            random_state=42,
            biased=True,
        )
        algo_svd.fit(trainset_svd)

    song_info_dict = _build_cf_song_info_dict(df_cf)

    return {
        'df_cf': df_z,
        'usercf': algo_usercf,
        'itemcf': algo_itemcf,
        'svd': algo_svd,
        'song_info_dict': song_info_dict,
    }


def _cf_predict_topn(algo_name, user_id, topk=10, user_history=None, train_rows_override=None):
    # 针对指定用户生成CF推荐
    history = _normalize_user_history(user_history)
    payload = (
        _get_personalized_cf_models(user_id, tuple(history), algo_name, train_rows_override)
        if history else _get_cf_models()
    )
    df_cf = payload['df_cf']
    algo = payload.get(algo_name)
    if algo is None:
        return []
    song_info_dict = payload['song_info_dict']

    user_id = int(user_id)
    all_songs = df_cf['song'].unique()
    user_listened = set(df_cf[df_cf['user'] == user_id]['song'].tolist())
    to_predict = [song for song in all_songs if song not in user_listened]

    # 样本内每首歌的偏好强度合计（列名为 play_count，实为 z），用于候选截断与分数相同时的次序
    song_strength = df_cf.groupby('song')['play_count'].sum().to_dict()

    # 控制候选集合：优先在「样本内更热」的歌上算分，避免 unique 顺序随机导致结果飘
    candidate_limit = min(1000, len(to_predict))
    to_predict = sorted(
        to_predict,
        key=lambda s: song_strength.get(s, 0.0),
        reverse=True,
    )[:candidate_limit]

    recommendations = []
    for song in to_predict:
        pred = algo.predict(user_id, song, clip=False)
        est = float(pred.est)
        if not np.isfinite(est):
            est = float("-inf")
        recommendations.append((song, est))

    # 先按推荐分排序，并按阈值过滤低分候选，避免出现明显不相关歌曲
    recommendations.sort(
        key=lambda x: (x[1], song_strength.get(x[0], 0.0)),
        reverse=True,
    )
    filtered = [x for x in recommendations if x[1] >= float(CF_MIN_EST_SCORE)]
    top_block = filtered[:topk]
    if not top_block:
        return []
    lines = []
    raw_vals = [float(raw) for _, raw in top_block]
    show_vals = _smooth_scale_scores(raw_vals, out_lo=0.0, out_hi=100.0)
    for i, (song, raw) in enumerate(top_block):
        dscore = show_vals[i]
        lines.append(
            f"{song_info_dict.get(song, '未知')} (song_id={song})，推荐值 {_display_float(dscore)}，原始估计 {_display_float(raw)}"
        )
    return lines


def usercf_topn(user_id, topk=10, user_history=None, train_rows_override=None):
    return _cf_predict_topn('usercf', user_id, topk, user_history=user_history, train_rows_override=train_rows_override)


def itemcf_topn(user_id, topk=10, user_history=None, train_rows_override=None):
    return _cf_predict_topn('itemcf', user_id, topk, user_history=user_history, train_rows_override=train_rows_override)


def svd_topn(user_id, topk=10, user_history=None, train_rows_override=None):
    return _cf_predict_topn('svd', user_id, topk, user_history=user_history, train_rows_override=train_rows_override)


@st.cache_resource
def _get_ncf_runtime():
    # 懒加载并缓存NCF模型和元数据
    ncf_df = pd.read_csv(
        DATA_FILE,
        nrows=NCF_SAMPLE_SIZE,
        usecols=['user', 'song'],
        dtype={'user': 'int32', 'song': 'int32'},
    )
    ext = _load_registered_history_interactions()[["user", "song"]]
    if not ext.empty:
        ncf_df = pd.concat([ncf_df, ext.astype({"user": "int32", "song": "int32"})], ignore_index=True).drop_duplicates()
    num_users = int(ncf_df['user'].max()) + 1
    num_songs = int(ncf_df['song'].max()) + 1
    model = None
    user2idx: dict = {}
    song2idx: dict = {}
    try:
        checkpoint = _load_torch_state_dict(NCF_MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 新格式：含密集编码映射表
            user2idx = _normalize_idx_map(checkpoint.get('user2idx', {}))
            song2idx = _normalize_idx_map(checkpoint.get('song2idx', {}))
            num_users = int(checkpoint.get('num_users', len(user2idx) or 1))
            num_songs = int(checkpoint.get('num_songs', len(song2idx) or 1))
            state_dict = checkpoint['state_dict']
        else:
            # 旧格式：纯 state_dict（向后兼容）
            state_dict = checkpoint
            num_users = int(state_dict['user_emb.weight'].shape[0])
            num_songs = int(state_dict['song_emb.weight'].shape[0])
        ckpt_dim = int(state_dict['user_emb.weight'].shape[1])

        # 兼容历史/异常 checkpoint：若映射表缺失，尝试按训练同口径重建映射
        # 训练时采用「排序后的唯一 user/song」做密集编码，这里用同规则恢复。
        if not user2idx or not song2idx:
            try:
                uniq_users = np.array(sorted(ncf_df["user"].astype(int).unique()), dtype=np.int64)
                uniq_songs = np.array(sorted(ncf_df["song"].astype(int).unique()), dtype=np.int64)
                if int(len(uniq_users)) == int(num_users) and int(len(uniq_songs)) == int(num_songs):
                    user2idx = {int(u): i for i, u in enumerate(uniq_users)}
                    song2idx = {int(s): i for i, s in enumerate(uniq_songs)}
            except Exception:
                # 重建失败时保持空映射，后续由评测层给出覆盖率提示
                user2idx = user2idx or {}
                song2idx = song2idx or {}

        model = NCF(num_users, num_songs, emb_dim=ckpt_dim)
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        model = None

    ncf_song_info = pd.read_csv(
        DATA_FILE, usecols=['song', 'artist_name', 'title']
    ).drop_duplicates('song').set_index('song')

    return {
        'ncf_df': ncf_df,
        'num_users': num_users,
        'num_songs': num_songs,
        'model': model,
        'song_info': ncf_song_info,
        'user2idx': user2idx,
        'song2idx': song2idx,
    }

def train_personalized_ncf(user_id, user_history, topk=10, n_epochs=3):
    # 基于用户历史记录训练个性化NCF模型
    if not user_history:
        return []
    
    try:
        ncf_payload = _get_ncf_runtime()
        ncf_song_info = ncf_payload['song_info']

        # 个性化阶段直接读取随机窗口 5w 行，避免先读 50w 造成响应变慢
        df_global = _read_random_ncf_window(
            PERSONALIZED_NCF_GLOBAL_SAMPLE,
            seed=int(user_id) % 10007,
        )
        ext = _load_registered_history_interactions()[["user", "song", "play_count"]]
        if not ext.empty:
            df_global = pd.concat(
                [df_global, ext.astype({"user": "int32", "song": "int32", "play_count": "float32"})],
                ignore_index=True,
            )
        
        # 创建用户历史数据（正反馈）：同一歌曲多次出现会提升采样权重
        hist_counts = pd.Series(user_history, dtype="int64").value_counts()
        user_data = pd.DataFrame({
            'user': [int(user_id)] * len(hist_counts),
            'song': hist_counts.index.astype("int64").tolist(),
            'play_count': (np.log1p(hist_counts.values.astype(np.float64)) + 1.0).astype(np.float32),
        })
        
        # 合并用户数据和全局随机样本
        df_train = pd.concat([user_data, df_global], ignore_index=True)
        df_train["play_count"] = pd.to_numeric(df_train["play_count"], errors="coerce").fillna(1.0).clip(lower=1e-6)
        df_train = df_train.groupby(["user", "song"], as_index=False)["play_count"].max()
        
        # 密集 ID 编码：避免稀疏原始 ID 直接作为 embedding 索引
        unique_users = np.array(sorted(df_train['user'].astype(int).unique()), dtype=np.int64)
        unique_songs = np.array(sorted(df_train['song'].astype(int).unique()), dtype=np.int64)
        user2idx = {int(u): i for i, u in enumerate(unique_users)}
        song2idx = {int(s): i for i, s in enumerate(unique_songs)}
        df_train = df_train.copy()
        df_train['user'] = df_train['user'].astype(int).map(user2idx).astype(np.int32)
        df_train['song'] = df_train['song'].astype(int).map(song2idx).astype(np.int32)
        user_enc = user2idx.get(int(user_id))
        if user_enc is None:
            return []
        num_users = len(unique_users)
        num_songs = len(unique_songs)
        
        # 与预训练主模型保持同一结构，避免覆盖保存后在其他模块加载失败
        model = NCF(num_users, num_songs, emb_dim=64)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-6)

        user_pos, pos_arrays, pos_probs = _build_user_sampling_tables(df_train, weighted_pos=True)
        users_arr = np.array(list(user_pos.keys()), dtype=np.int64)
        song_ids = np.arange(num_songs, dtype=np.int64)
        song_sampling_probs = _build_song_sampling_probs(df_train, num_songs)
        rng = np.random.default_rng(42)
        batch_size = min(256, max(64, len(df_train) // 6))
        steps_per_epoch = max(30, len(df_train) // max(batch_size, 1))
        scaler = _make_grad_scaler(device)
        
        model.train()
        for epoch in range(n_epochs):
            for _ in range(steps_per_epoch):
                optimizer.zero_grad()
                users, pos_items, neg_items = _sample_bpr_triplets(
                    user_pos=user_pos,
                    all_users=users_arr,
                    num_songs=num_songs,
                    batch_size=batch_size,
                    rng=rng,
                    pos_arrays=pos_arrays,
                    pos_probs=pos_probs,
                    song_sampling_probs=song_sampling_probs,
                    song_ids=song_ids,
                    neg_k=3,
                    popular_negative_ratio=0.67,
                )
                user_t = torch.tensor(users, dtype=torch.long, device=device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
                with _autocast_ctx(device):
                    pos_score = model(user_t, pos_t)
                    if neg_items.ndim == 1:
                        neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)
                        neg_score = model(user_t, neg_t)
                        loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
                    else:
                        bsz, kneg = int(neg_items.shape[0]), int(neg_items.shape[1])
                        neg_t = torch.tensor(neg_items.reshape(-1), dtype=torch.long, device=device)
                        user_rep = user_t.repeat_interleave(kneg)
                        neg_score = model(user_rep, neg_t).view(bsz, kneg)
                        pos_rep = pos_score.unsqueeze(1).expand_as(neg_score)
                        loss = -torch.nn.functional.logsigmoid(pos_rep - neg_score).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        
        # 个性化模型单独保存，避免覆盖全局 NCF（算法对比页读取全局模型）
        os.makedirs(os.path.dirname(NCF_PERSONALIZED_MODEL_PATH), exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "user2idx": user2idx,
                "song2idx": song2idx,
                "num_users": num_users,
                "num_songs": num_songs,
            },
            NCF_PERSONALIZED_MODEL_PATH,
        )

        # 生成推荐
        model.eval()
        all_songs = df_global['song'].astype(int).unique().tolist()
        user_listened = set(int(s) for s in user_history)
        candidate_songs = [int(s) for s in all_songs if int(s) not in user_listened and int(s) in song2idx]
        
        if len(candidate_songs) == 0:
            return []
        
        # 预测候选歌曲分数
        candidate_enc = [song2idx[int(s)] for s in candidate_songs]
        user_tensor = torch.tensor([int(user_enc)] * len(candidate_enc), dtype=torch.long).to(device)
        song_tensor = torch.tensor(candidate_enc, dtype=torch.long).to(device)
        
        with torch.no_grad():
            scores = model(user_tensor, song_tensor).cpu().numpy()
        
        # 排序并取TopK
        song_score_pairs = list(zip(candidate_songs, scores))
        song_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_recommend = song_score_pairs[:topk]
        show_vals = _smooth_scale_scores([float(s) for _, s in top_recommend], out_lo=0.0, out_hi=100.0)
        
        # 格式化结果
        result = []
        for i, (sid, score) in enumerate(top_recommend):
            if sid in ncf_song_info.index:
                artist = ncf_song_info.loc[sid, 'artist_name']
                title = ncf_song_info.loc[sid, 'title']
                result.append(
                    f"{artist} - {title} (song_id={sid})，推荐值 {_display_float(show_vals[i])}"
                )
            else:
                result.append(
                    f"song_id={sid}，推荐值 {_display_float(show_vals[i])}"
                )
        
        return result
        
    except Exception as e:
        # 训练失败，返回空列表
        import traceback
        print(f"个性化训练失败: {str(e)}")
        print(traceback.format_exc())
        return []


def ncf_recommend(user_id, topk=10, user_history=None):
    # NCF深度学习推荐TopN
    # 只要有历史记录，就按"用户数据并入训练集后重训并覆盖模型"的策略执行
    if user_history:
        return train_personalized_ncf(user_id, user_history, topk)

    payload = _get_ncf_runtime()
    num_songs = payload['num_songs']
    ncf_model = payload['model']
    ncf_song_info = payload['song_info']
    user2idx: dict = payload.get('user2idx', {})
    song2idx: dict = payload.get('song2idx', {})
    idx2song: dict = {v: k for k, v in song2idx.items()} if song2idx else {}

    if ncf_model is None:
        raise FileNotFoundError(f"模型文件 {NCF_MODEL_PATH} 不存在，请先训练模型")

    # 把原始 user_id 编码（新格式有映射表；旧格式直接用原始 ID）
    if user2idx:
        user_enc = user2idx.get(int(user_id))
        if user_enc is None:
            return []
    else:
        user_enc = int(user_id)

    # 过滤已听歌曲（在原始 song_id 空间判断）
    seen_raw: set = set()
    ncf_df = payload["ncf_df"]
    seen_mask = ncf_df["user"].astype("int64") == int(user_id)
    if bool(seen_mask.any()):
        seen_raw = set(ncf_df["song"][seen_mask].astype("int64").tolist())

    # 候选集：所有编码后的 song_enc，过滤已听的原始 song_id
    if song2idx:
        candidate_encs = np.array(
            [enc for raw_sid, enc in song2idx.items() if raw_sid not in seen_raw],
            dtype=np.int64,
        )
    else:
        candidate_encs = np.array(
            [sid for sid in range(num_songs) if sid not in seen_raw],
            dtype=np.int64,
        )

    if candidate_encs.size == 0:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncf_model = ncf_model.to(device)
    ncf_model.eval()
    batch = 65536 if device.type == "cuda" else 16384
    score_blocks = []
    with torch.no_grad():
        for start in range(0, candidate_encs.size, batch):
            part = candidate_encs[start:start + batch]
            u_t = torch.full((len(part),), user_enc, dtype=torch.long, device=device)
            s_t = torch.tensor(part, dtype=torch.long, device=device)
            out = ncf_model(u_t, s_t).detach().cpu().numpy()
            score_blocks.append(out)

    scores = np.concatenate(score_blocks, axis=0)
    k = min(int(topk), int(candidate_encs.size))
    if k <= 0:
        return []
    top_local = np.argpartition(-scores, kth=k - 1)[:k]
    top_sorted = top_local[np.argsort(-scores[top_local])]
    top_encs = candidate_encs[top_sorted]
    top_scores = scores[top_sorted]
    top_raw_sids = [idx2song.get(int(enc), int(enc)) for enc in top_encs]

    show_vals = _smooth_scale_scores([float(x) for x in top_scores], out_lo=0.0, out_hi=100.0)
    result = []
    for i, sid in enumerate(top_raw_sids):
        if sid in ncf_song_info.index:
            artist = ncf_song_info.loc[sid, 'artist_name']
            title = ncf_song_info.loc[sid, 'title']
            result.append(
                f"{artist} - {title} (song_id={sid})，推荐值 {_display_float(show_vals[i])}"
            )
        else:
            result.append(
                f"song_id={sid}，推荐值 {_display_float(show_vals[i])}"
            )
    return result

@st.cache_data
def _get_content_features():
    # 缓存内容特征；仅用前若干行构建歌曲表，显著快于全表扫描
    content_features = ["artist_name", "title", "release", "year", "artist_familiarity", "artist_hotttnesss"]
    try:
        df = pd.read_csv(
            DATA_FILE,
            usecols=["song"] + content_features,
            nrows=max(CONTENT_SAMPLE_SIZE, STREAMLIT_DATA_NROWS // 2),
        )
    except ValueError:
        # 兼容缺少 release 列的数据集
        fallback_features = ["artist_name", "title", "year", "artist_familiarity", "artist_hotttnesss"]
        df = pd.read_csv(
            DATA_FILE,
            usecols=["song"] + fallback_features,
            nrows=max(CONTENT_SAMPLE_SIZE, STREAMLIT_DATA_NROWS // 2),
        )
        df["release"] = "unknown"
    songs = df.drop_duplicates("song")[["song"] + content_features].set_index("song")
    return songs

@st.cache_data
def _get_feature_encoder():
    # 缓存特征编码器
    cached = _safe_pickle_load(CONTENT_MODEL_BUNDLE_PATH)
    if isinstance(cached, dict) and cached.get("meta", {}).get("content_sample_size") == CONTENT_SAMPLE_SIZE:
        if {"artist_hasher", "title_hasher", "release_hasher", "scaler", "songs"}.issubset(set(cached.keys())):
            if int(cached.get("meta", {}).get("feature_version", 1)) >= 2:
                return (
                    cached["artist_hasher"],
                    cached["title_hasher"],
                    cached["release_hasher"],
                    cached["scaler"],
                    cached["songs"],
                )

    songs = _get_content_features()
    # 创建特征编码器
    artist_hasher = FeatureHasher(n_features=32, input_type='string')
    title_hasher = FeatureHasher(n_features=64, input_type='string')
    release_hasher = FeatureHasher(n_features=32, input_type='string')
    scaler = MinMaxScaler()
    
    # 准备数据
    artist_names = songs['artist_name'].fillna('unknown').astype(str).values
    title_names = songs['title'].fillna('unknown').astype(str).values
    release_names = songs['release'].fillna('unknown').astype(str).values
    num_features = songs[['year', 'artist_familiarity', 'artist_hotttnesss']].fillna(0).values
    
    # 训练编码器
    artist_list = [[name] for name in artist_names]
    title_list = [[name] for name in title_names]
    release_list = [[name] for name in release_names]
    artist_hasher.transform(artist_list)
    title_hasher.transform(title_list)
    release_hasher.transform(release_list)
    scaler.fit_transform(num_features)
    
    _safe_pickle_dump(
        CONTENT_MODEL_BUNDLE_PATH,
        {
            "artist_hasher": artist_hasher,
            "title_hasher": title_hasher,
            "release_hasher": release_hasher,
            "scaler": scaler,
            "songs": songs,
            "meta": {
                "content_sample_size": CONTENT_SAMPLE_SIZE,
                "n_songs": int(len(songs)),
                "feature_version": 2,
            },
        },
    )
    return artist_hasher, title_hasher, release_hasher, scaler, songs

def _get_song_feature(song_id, artist_hasher, title_hasher, release_hasher, scaler, songs):
    # 获取单个歌曲的特征向量
    if song_id not in songs.index:
        return None
    
    row = songs.loc[song_id]
    artist_name = str(row['artist_name']) if pd.notna(row['artist_name']) else 'unknown'
    title_name = str(row['title']) if pd.notna(row['title']) else 'unknown'
    release_name = str(row['release']) if pd.notna(row['release']) else 'unknown'
    artist_feature = artist_hasher.transform([[artist_name]]).toarray()[0]
    title_feature = title_hasher.transform([[title_name]]).toarray()[0]
    release_feature = release_hasher.transform([[release_name]]).toarray()[0]
    
    num_values = [[row['year'] if pd.notna(row['year']) else 0,
                   row['artist_familiarity'] if pd.notna(row['artist_familiarity']) else 0,
                   row['artist_hotttnesss'] if pd.notna(row['artist_hotttnesss']) else 0]]
    num_feature = scaler.transform(num_values)[0]
    
    return np.hstack([artist_feature, title_feature, release_feature, num_feature])


@st.cache_data
def _get_content_song_matrix():
    """缓存内容向量矩阵，避免每次逐首重复编码。"""
    artist_hasher, title_hasher, release_hasher, scaler, songs = _get_feature_encoder()
    song_ids = songs.index.astype(int).tolist()
    if not song_ids:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=object), np.zeros((0,), dtype=np.float32), songs

    valid_song_ids = []
    mat = []
    for sid in song_ids:
        vec = _get_song_feature(sid, artist_hasher, title_hasher, release_hasher, scaler, songs)
        if vec is None:
            continue
        valid_song_ids.append(int(sid))
        mat.append(np.asarray(vec, dtype=np.float32))
    if not mat:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=object), np.zeros((0,), dtype=np.float32), songs

    song_ids_arr = np.asarray(valid_song_ids, dtype=np.int64)
    song_mat = np.vstack(mat).astype(np.float32)
    song_norm = np.linalg.norm(song_mat, axis=1).astype(np.float32) + 1e-8
    artist_arr = songs.loc[song_ids_arr, "artist_name"].fillna("unknown").astype(str).values
    year_arr = pd.to_numeric(songs.loc[song_ids_arr, "year"], errors="coerce").fillna(0).astype(np.float32).values
    return song_ids_arr, song_mat, song_norm, artist_arr, year_arr, songs

def content_based_recommend(user_history_songs, topk=10):
    # 基于用户历史的内容推荐
    if not user_history_songs:
        return []

    song_ids_arr, song_mat, song_norm, artist_arr, year_arr, songs = _get_content_song_matrix()
    if song_mat.size == 0:
        return []

    sid_to_idx = {int(sid): i for i, sid in enumerate(song_ids_arr.tolist())}
    history_idx = [sid_to_idx[int(sid)] for sid in user_history_songs if int(sid) in sid_to_idx]
    if not history_idx:
        return []

    user_profile = np.mean(song_mat[history_idx], axis=0)
    profile_norm = float(np.linalg.norm(user_profile) + 1e-8)
    similarity = (song_mat @ user_profile) / (song_norm * profile_norm)

    listened = {int(sid) for sid in user_history_songs}
    keep_mask = np.array([int(sid) not in listened for sid in song_ids_arr], dtype=bool)
    if not np.any(keep_mask):
        return []

    # 内容互补增强：强调「历史主流歌手」和「偏好年代带」，不依赖时间戳或流行度先验
    hist_artist = artist_arr[history_idx]
    top_artists = set(pd.Series(hist_artist).value_counts().head(3).index.tolist())
    artist_bonus = np.where(np.isin(artist_arr, list(top_artists)), 0.08, 0.0).astype(np.float32)

    hist_year = year_arr[history_idx]
    valid_hist_year = hist_year[hist_year > 0]
    if len(valid_hist_year) > 0:
        target_year = float(np.median(valid_hist_year))
        year_affinity = np.clip(1.0 - np.abs(year_arr - target_year) / 20.0, 0.0, 1.0).astype(np.float32)
        year_bonus = 0.05 * year_affinity
    else:
        year_bonus = np.zeros_like(similarity, dtype=np.float32)

    final_score = similarity.astype(np.float32) + artist_bonus + year_bonus
    final_score = np.where(keep_mask, final_score, -np.inf)

    eff_topk = int(max(1, min(int(topk), int(np.sum(keep_mask)))))
    top_idx = np.argpartition(final_score, -eff_topk)[-eff_topk:]
    top_idx = top_idx[np.argsort(final_score[top_idx])[::-1]]

    result = []
    show_vals = _smooth_scale_scores([float(final_score[i]) for i in top_idx], out_lo=0.0, out_hi=100.0)
    for rank, idx in enumerate(top_idx):
        song_id = int(song_ids_arr[idx])
        pure_sim = float(similarity[idx])
        score = float(final_score[idx])
        row = songs.loc[song_id]
        result.append(
            f"{row['artist_name']} - {row['title']} (song_id={song_id})，"
            f"相似度 {_display_float(pure_sim)}，"
            f"融合分 {_display_float(score)}，推荐值 {_display_float(show_vals[rank])}"
        )
    return result


def extract_song_id_from_result(result_str):
    # 从推荐结果字符串中提取song_id
    import re
    match = re.search(r'song_id=(\d+)', result_str)
    if match:
        return int(match.group(1))
    return None


def get_max_user_id_for_hybrid():
    # 获取数据中的最大用户ID
    try:
        return int(_get_ncf_runtime()['num_users'] - 1)
    except Exception:
        return 1000


HYBRID_WEIGHT_KEYS = ("usercf", "itemcf", "svd", "ncf", "content")


def default_hybrid_weights_5way(user_id, user_history):
    """无自定义权重时的五路默认权重（已归一化）。按历史量与是否在 NCF 用户表内微调。"""
    h = len(user_history) if user_history else 0
    max_uid = int(get_max_user_id_for_hybrid())
    in_ncf = user_id is not None and int(user_id) <= max_uid
    # 新默认：当 NCF 明显领先时，融合权重向 NCF 倾斜
    w = {"usercf": 0.20, "itemcf": 0.15, "svd": 0.15, "ncf": 0.40, "content": 0.10}
    if h < 5:
        w = {"usercf": 0.18, "itemcf": 0.14, "svd": 0.14, "ncf": 0.36, "content": 0.18}
    if h == 0:
        w = {"usercf": 0.14, "itemcf": 0.12, "svd": 0.14, "ncf": 0.34, "content": 0.26}
    if not in_ncf:
        n = float(w["ncf"])
        w["ncf"] = 0.0
        w["content"] += n * 0.45
        w["svd"] += n * 0.28
        w["usercf"] += n * 0.135
        w["itemcf"] += n * 0.135
    if not user_history:
        w["content"] *= 0.35
        w["svd"] += 0.12
        w["usercf"] += 0.1
        w["itemcf"] += 0.1
    s = sum(max(0.0, float(v)) for v in w.values())
    if s < 1e-12:
        return {k: 0.2 for k in HYBRID_WEIGHT_KEYS}
    return {k: max(0.0, float(w[k])) / s for k in HYBRID_WEIGHT_KEYS}


def coalesce_hybrid_weights(weights, user_id, user_history):
    """
    归一为五路权重 dict。兼容旧版 {'cf','ncf','content'}：将 cf 均分到 usercf、itemcf、svd 后再与 ncf、content 合并归一。
    weights 为 None 时用 default_hybrid_weights_5way。
    """
    if weights is None:
        return default_hybrid_weights_5way(user_id, user_history)
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in HYBRID_WEIGHT_KEYS}
    cf_legacy = max(0.0, float(weights.get("cf", 0.0)))
    if cf_legacy > 0:
        w["usercf"] += cf_legacy / 3.0
        w["itemcf"] += cf_legacy / 3.0
        w["svd"] += cf_legacy / 3.0
    s = sum(w.values())
    if s < 1e-12:
        return default_hybrid_weights_5way(user_id, user_history)
    return {k: w[k] / s for k in HYBRID_WEIGHT_KEYS}


def _parse_cf_raw_estimate(result_str: str):
    import re

    m = re.search(r"原始估计\s+(.+?)(?:，|$)", result_str)
    if not m:
        m = re.search(r"推荐值\s+(.+?)(?:，|$)", result_str)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except ValueError:
        return None


def _parse_ncf_raw(result_str: str):
    import re

    m = re.search(r"原始分\s+(.+?)(?:，|$)", result_str)
    if not m:
        m = re.search(r"推荐值\s+(.+?)(?:，|$)", result_str)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except ValueError:
        return None


def _parse_content_fusion_score(result_str: str):
    """与内容页排序一致：优先用「融合分」（相似度与全站先验加权），否则退回相似度。"""
    import re

    m = re.search(r"融合分\s+(.+?)(?:，|$)", result_str)
    if not m:
        m = re.search(r"相似度\s+(.+?)(?:，|$)", result_str)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except ValueError:
        return None


def hybrid_recommend(user_id, user_history, topk=10, weights=None):
    # 五路加权融合：UserCF / ItemCF / SVD / NCF / 内容；各路分数在候选并集上分别校准后再加权
    import re

    w = coalesce_hybrid_weights(weights, user_id, user_history)
    cand_n = max(topk * 2, 40)
    song_scores = {}

    def _ingest_cf_rows(results, key, str_key):
        for result_str in results or []:
            song_id = extract_song_id_from_result(result_str)
            if song_id is None:
                continue
            score = _parse_cf_raw_estimate(result_str)
            if score is None:
                continue
            if song_id not in song_scores:
                song_scores[song_id] = {}
            song_scores[song_id][key] = score
            song_scores[song_id][str_key] = result_str

    if w.get("usercf", 0) > 0:
        try:
            _ingest_cf_rows(usercf_topn(user_id, cand_n, user_history=user_history), "usercf", "usercf_str")
        except Exception:
            pass
    if w.get("itemcf", 0) > 0:
        try:
            _ingest_cf_rows(itemcf_topn(user_id, cand_n, user_history=user_history), "itemcf", "itemcf_str")
        except Exception:
            pass
    if w.get("svd", 0) > 0:
        try:
            _ingest_cf_rows(svd_topn(user_id, cand_n, user_history=user_history), "svd", "svd_str")
        except Exception:
            pass

    if w.get("ncf", 0) > 0:
        try:
            for result_str in ncf_recommend(user_id, cand_n, user_history) or []:
                song_id = extract_song_id_from_result(result_str)
                if song_id is None:
                    continue
                score = _parse_ncf_raw(result_str)
                if score is None:
                    continue
                if song_id not in song_scores:
                    song_scores[song_id] = {}
                song_scores[song_id]["ncf"] = score
                song_scores[song_id]["ncf_str"] = result_str
        except Exception:
            pass

    if w.get("content", 0) > 0 and user_history:
        try:
            for result_str in content_based_recommend(user_history, cand_n) or []:
                song_id = extract_song_id_from_result(result_str)
                if song_id is None:
                    continue
                score = _parse_content_fusion_score(result_str)
                if score is None:
                    continue
                if song_id not in song_scores:
                    song_scores[song_id] = {}
                song_scores[song_id]["content"] = score
                song_scores[song_id]["content_str"] = result_str
        except Exception:
            pass

    def _calibrate_field(field):
        vals = [song_scores[sid][field] for sid in song_scores if field in song_scores[sid]]
        if not vals:
            return
        cals = _calibrate_scores(vals)
        i = 0
        for sid in song_scores:
            if field in song_scores[sid]:
                song_scores[sid][field] = cals[i]
                i += 1

    for _field in HYBRID_WEIGHT_KEYS:
        _calibrate_field(_field)

    fusion_scores = []
    for song_id, scores in song_scores.items():
        fusion_score = 0.0
        used_weight = 0.0
        for key in HYBRID_WEIGHT_KEYS:
            if key not in scores:
                continue
            wt = float(w.get(key, 0.0))
            if wt <= 0:
                continue
            fusion_score += wt * scores[key]
            used_weight += wt
        if used_weight > 1e-8:
            fusion_score = fusion_score / used_weight

        song_info = (
            scores.get("usercf_str")
            or scores.get("itemcf_str")
            or scores.get("svd_str")
            or scores.get("content_str")
            or scores.get("ncf_str")
            or f"song_id={song_id}"
        )
        info_match = re.search(r"(.+?)\s*\(song_id=", song_info)
        if info_match:
            song_name = info_match.group(1).strip()
        else:
            song_name = f"song_id={song_id}"

        fusion_scores.append((song_id, fusion_score, song_name))

    fusion_scores.sort(key=lambda x: x[1], reverse=True)
    top_recommend = fusion_scores[:topk]

    result = []
    for song_id, score, song_name in top_recommend:
        result.append(f"{song_name} (song_id={song_id})，综合分 {_display_float(score)}")
    return result
