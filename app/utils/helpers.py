# 工具函数模块
import os
import json
from datetime import datetime
import hashlib
import requests
import base64
import hashlib
import hmac
import pandas as pd
import streamlit as st
from config import (
    FONT_PATH,
    FONT_URL,
    DATA_FILE,
    HISTORY_FILE,
    MAX_HISTORY_SIZE,
    HISTORY_PAGE_SIZE,
    HISTORY_MAX_PAGES,
    NCF_SAMPLE_SIZE,
    STREAMLIT_DATA_NROWS,
    USERS_FILE,
    MODEL_DIR,
    CF_MODEL_BUNDLE_PATH,
    CF_USERCF_MODEL_PATH,
    CF_ITEMCF_MODEL_PATH,
    CF_SVD_MODEL_PATH,
    CF_META_PATH,
    CONTENT_MODEL_BUNDLE_PATH,
    NCF_MODEL_PATH,
    CF_SAMPLE_SIZE,
    CONTENT_SAMPLE_SIZE,
)

_DATA_ROWS_CACHE_SIG = None
_DATA_ROWS_CACHE_VAL = None


def _hash_password(password, salt_bytes=None):
    # 使用PBKDF2-HMAC-SHA256做密码哈希
    if salt_bytes is None:
        salt_bytes = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256', password.encode('utf-8'), salt_bytes, 120000
    )
    return (
        base64.b64encode(hashed).decode('utf-8'),
        base64.b64encode(salt_bytes).decode('utf-8')
    )


def _verify_password(password, password_hash, salt_b64):
    # 校验密码哈希
    try:
        salt_bytes = base64.b64decode(salt_b64.encode('utf-8'))
        computed_hash, _ = _hash_password(password, salt_bytes=salt_bytes)
        return hmac.compare_digest(computed_hash, password_hash)
    except Exception:
        return False


def download_font_if_needed():
    # 下载中文字体文件
    if not os.path.exists(FONT_PATH):
        try:
            st.info("正在下载中文字体文件...")
            r = requests.get(FONT_URL, timeout=10)
            r.raise_for_status()
            os.makedirs(os.path.dirname(FONT_PATH), exist_ok=True)
            with open(FONT_PATH, "wb") as f:
                f.write(r.content)
            st.success("字体文件下载完成！")
        except Exception as e:
            st.error(f"字体文件下载失败: {str(e)}")
            return None
    return FONT_PATH


@st.cache_data
def load_song_info():
    # 加载歌曲信息
    try:
        df = pd.read_csv(
            DATA_FILE,
            usecols=['song', 'artist_name', 'title'],
            nrows=STREAMLIT_DATA_NROWS,
        ).drop_duplicates('song').set_index('song')
        return df
    except FileNotFoundError:
        st.error(f"数据文件不存在: {DATA_FILE}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"加载数据文件失败: {str(e)}")
        return pd.DataFrame()


def extract_song_id(text):
    # 从文本中提取song_id
    import re
    match = re.search(r'song_id=(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def load_user_history_from_file(username):
    # 从文件加载用户历史记录
    if not username:
        return []
    
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
                h = all_history.get(username, [])
                if len(h) > MAX_HISTORY_SIZE:
                    return h[-MAX_HISTORY_SIZE:]
                return h
    except Exception as e:
        st.warning(f"加载历史记录失败: {str(e)}")
    return []


def save_user_history_to_file(username, history):
    # 保存用户历史记录到文件
    if not username:
        return
    
    try:
        # 读取所有用户的历史记录
        all_history = {}
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
        
        # 更新当前用户的历史记录
        all_history[username] = history
        
        # 保存到文件
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"保存历史记录失败: {str(e)}")


def save_to_history(song_ids, max_size=None):
    # 保存歌曲ID到用户历史记录
    if max_size is None:
        max_size = MAX_HISTORY_SIZE
    
    username = st.session_state.get('username', '')
    if not username:
        return
    
    if 'user_history' not in st.session_state:
        st.session_state['user_history'] = []
    
    # 添加新歌曲
    for song_id in song_ids:
        if song_id is not None and song_id not in st.session_state['user_history']:
            st.session_state['user_history'].append(song_id)
    
    # 只保留最近N条记录
    st.session_state['user_history'] = st.session_state['user_history'][-max_size:]
    
    # 保存到文件
    save_user_history_to_file(username, st.session_state['user_history'])


def remove_from_history(song_id):
    # 从历史记录中删除歌曲
    username = st.session_state.get('username', '')
    if not username:
        return False
    
    if 'user_history' not in st.session_state:
        return False
    
    if song_id in st.session_state['user_history']:
        st.session_state['user_history'].remove(song_id)
        # 保存到文件
        save_user_history_to_file(username, st.session_state['user_history'])
        return True
    return False


@st.cache_data
def get_max_user_id():
    # 获取数据中的最大用户ID
    try:
        df = pd.read_csv(DATA_FILE, usecols=['user'], nrows=NCF_SAMPLE_SIZE)
        return int(df['user'].max())
    except:
        return 1000  # 默认值


def get_data_total_rows():
    """统计主数据文件总行数（不含表头），并按文件签名缓存。"""
    global _DATA_ROWS_CACHE_SIG, _DATA_ROWS_CACHE_VAL
    try:
        st_info = os.stat(DATA_FILE)
        sig = (int(st_info.st_mtime_ns), int(st_info.st_size))
        if _DATA_ROWS_CACHE_SIG == sig and _DATA_ROWS_CACHE_VAL is not None:
            return int(_DATA_ROWS_CACHE_VAL)

        with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
            next(f, None)  # 跳过表头
            cnt = sum(1 for _ in f)
        rows = max(int(cnt), 1000)
        _DATA_ROWS_CACHE_SIG = sig
        _DATA_ROWS_CACHE_VAL = rows
        return rows
    except Exception:
        # 兜底返回当前配置规模
        return max(int(NCF_SAMPLE_SIZE), 1000)


def load_users():
    # 加载所有用户注册信息
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"加载用户信息失败: {str(e)}")
    return {}


def save_users(users_data):
    # 保存用户注册信息
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存用户信息失败: {str(e)}")
        return False


def register_user(username, password):
    # 注册新用户
    if not username or not password:
        return False, "用户名和密码不能为空"
    
    users = load_users()
    
    # 检查用户名是否已存在
    if username in users:
        return False, "用户名已存在，请选择其他用户名"
    
    # 获取下一个可用的用户ID
    max_user_id = get_max_user_id()
    existing_ids = {user_data.get('user_id', 0) for user_data in users.values()}
    
    # 分配新用户ID
    new_user_id = max_user_id + 1
    while new_user_id in existing_ids:
        new_user_id += 1
    
    # 保存用户信息
    from datetime import datetime
    password_hash, password_salt = _hash_password(password)

    users[username] = {
        'user_id': new_user_id,
        'password_hash': password_hash,
        'password_salt': password_salt,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if save_users(users):
        return True, f"注册成功！你的用户ID是 {new_user_id}"
    else:
        return False, "注册失败，请重试"


def verify_user(username, password):
    # 验证用户登录
    if not username or not password:
        return False, None, "用户名和密码不能为空"
    
    users = load_users()
    
    if username not in users:
        return False, None, "用户名不存在"
    
    user_data = users[username]
    # 新版：哈希密码验证
    stored_hash = user_data.get('password_hash')
    stored_salt = user_data.get('password_salt')
    if stored_hash and stored_salt:
        if not _verify_password(password, stored_hash, stored_salt):
            return False, None, "密码错误"
    else:
        # 兼容旧版明文存储，首次登录后自动迁移
        if user_data.get('password') != password:
            return False, None, "密码错误"
        new_hash, new_salt = _hash_password(password)
        user_data['password_hash'] = new_hash
        user_data['password_salt'] = new_salt
        user_data.pop('password', None)
        save_users(users)
    
    user_id = user_data.get('user_id')
    return True, user_id, "登录成功"


def get_user_id_by_username(username):
    # 根据用户名获取用户ID
    users = load_users()
    if username in users:
        return users[username].get('user_id')
    return None

def history_storage_to_newest_first(song_ids):
    """存储顺序为旧→新，展示新→旧时反转。"""
    return list(reversed(list(song_ids or [])))


def history_num_pages(n_total: int) -> int:
    if n_total <= 0:
        return 0
    return min(HISTORY_MAX_PAGES, (n_total + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)


def clamp_history_view_page():
    """历史条数变化后，校正当前页码。"""
    uh = st.session_state.get("user_history", [])
    n = len(uh)
    npages = history_num_pages(n)
    cur = int(st.session_state.get("history_view_page", 0))
    if npages <= 0:
        st.session_state["history_view_page"] = 0
        return 0, 0, n
    if cur >= npages:
        st.session_state["history_view_page"] = npages - 1
        cur = npages - 1
    return cur, npages, n


def get_history_page_slice(page_idx: int):
    """
    返回当前页展示：新→旧。
    Returns:
        (display_lines: list[str], song_ids: list[int])
    """
    uh = st.session_state.get("user_history", [])
    rev = history_storage_to_newest_first(uh)
    n = len(rev)
    npages = history_num_pages(n)
    if n == 0 or npages == 0:
        return [], []
    page_idx = max(0, min(int(page_idx), npages - 1))
    start = page_idx * HISTORY_PAGE_SIZE
    chunk = rev[start : start + HISTORY_PAGE_SIZE]
    df_hist = load_song_info()
    lines = []
    for sid in chunk:
        if not df_hist.empty and sid in df_hist.index:
            artist = df_hist.loc[sid, "artist_name"]
            title = df_hist.loc[sid, "title"]
            lines.append(f"{artist} - {title} (song_id={sid})")
        else:
            lines.append(f"song_id={sid}")
    return lines, chunk


def get_user_history_display():
    """全量历史，存储顺序旧→新。分页展示请用 get_history_page_slice。"""
    if "user_history" not in st.session_state or not st.session_state["user_history"]:
        return []

    df_hist = load_song_info()
    if df_hist.empty:
        return [f"song_id={sid}" for sid in st.session_state["user_history"]]

    result = []
    for sid in st.session_state["user_history"]:
        if sid in df_hist.index:
            artist = df_hist.loc[sid, "artist_name"]
            title = df_hist.loc[sid, "title"]
            result.append(f"{artist} - {title} (song_id={sid})")
        else:
            result.append(f"song_id={sid}")
    return result


def _file_sig(path):
    if not os.path.exists(path):
        return [False, 0, ""]
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if not chunk:
                break
            hasher.update(chunk)
    stt = os.stat(path)
    return [True, int(stt.st_size), hasher.hexdigest()]


def _current_user_data_signature():
    return {
        "users": _file_sig(USERS_FILE),
        "history": _file_sig(HISTORY_FILE),
    }


def _normalize_sig(obj):
    if isinstance(obj, tuple):
        return [_normalize_sig(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize_sig(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_sig(v) for k, v in obj.items()}
    return obj


def _training_meta_path():
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, "training_meta.json")


def _load_training_meta():
    path = _training_meta_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_training_meta(meta):
    path = _training_meta_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_last_model_train_params():
    """
    返回各模型最近一次重训参数（若无则回退到当前配置默认值）。
    """
    meta = _load_training_meta()
    p = meta.get("last_train_params", {}) if isinstance(meta, dict) else {}
    cf_meta = {}
    if os.path.isfile(CF_META_PATH):
        try:
            with open(CF_META_PATH, "r", encoding="utf-8") as f:
                cf_meta = json.load(f)
        except Exception:
            cf_meta = {}

    cf_u_meta = int(cf_meta.get("cf_usercf_sample_size", CF_SAMPLE_SIZE)) if isinstance(cf_meta, dict) else int(CF_SAMPLE_SIZE)
    cf_i_meta = int(cf_meta.get("cf_itemcf_sample_size", CF_SAMPLE_SIZE)) if isinstance(cf_meta, dict) else int(CF_SAMPLE_SIZE)
    cf_s_meta = int(cf_meta.get("cf_svd_sample_size", CF_SAMPLE_SIZE)) if isinstance(cf_meta, dict) else int(CF_SAMPLE_SIZE)
    cf_rows_meta = max(cf_u_meta, cf_i_meta, cf_s_meta)

    cf_rows = int(p.get("cf_rows", cf_rows_meta))
    content_rows = int(p.get("content_rows", CONTENT_SAMPLE_SIZE))
    ncf_rows = int(p.get("ncf_rows", NCF_SAMPLE_SIZE))
    ncf_epochs = int(p.get("ncf_epochs", 8))
    ncf_lr = float(p.get("ncf_lr", 0.0006))
    ncf_emb_dim = int(p.get("ncf_emb_dim", 64))
    ncf_neg_k = int(p.get("ncf_neg_k", 3))
    ncf_pop_ratio = float(p.get("ncf_popular_negative_ratio", 0.67))
    ncf_min_epochs = int(p.get("ncf_min_epochs", 5))
    ncf_patience = int(p.get("ncf_patience", 4))
    ncf_batch_size = int(p.get("ncf_batch_size", 1024))
    usercf_rows = int(p.get("usercf_rows", cf_u_meta))
    itemcf_rows = int(p.get("itemcf_rows", cf_i_meta))
    svd_rows = int(p.get("svd_rows", cf_s_meta))
    return {
        "cf_rows": max(1000, cf_rows),
        "usercf_rows": max(1000, usercf_rows),
        "itemcf_rows": max(1000, itemcf_rows),
        "svd_rows": max(1000, svd_rows),
        "content_rows": max(1000, content_rows),
        "ncf_rows": max(1000, ncf_rows),
        "ncf_epochs": max(1, ncf_epochs),
        "ncf_lr": max(1e-6, ncf_lr),
        "ncf_emb_dim": max(8, ncf_emb_dim),
        "ncf_neg_k": max(1, ncf_neg_k),
        "ncf_popular_negative_ratio": min(1.0, max(0.0, ncf_pop_ratio)),
        "ncf_min_epochs": max(1, ncf_min_epochs),
        "ncf_patience": max(1, ncf_patience),
        "ncf_batch_size": max(64, ncf_batch_size),
    }


def persist_last_cf_train_params(usercf_rows: int, itemcf_rows: int, svd_rows: int) -> bool:
    """
    持久化协同过滤页最近一次使用的三路训练条数。
    用于页面重启后回填参数（即使本次未触发重训）。
    返回是否发生写入。
    """
    try:
        u = max(1000, int(usercf_rows))
        i = max(1000, int(itemcf_rows))
        s = max(1000, int(svd_rows))
    except Exception:
        return False

    meta = _load_training_meta()
    if not isinstance(meta, dict):
        meta = {}
    last_params = meta.get("last_train_params", {})
    if not isinstance(last_params, dict):
        last_params = {}

    changed = (
        int(last_params.get("usercf_rows", -1)) != u
        or int(last_params.get("itemcf_rows", -1)) != i
        or int(last_params.get("svd_rows", -1)) != s
        or int(last_params.get("cf_rows", -1)) != max(u, i, s)
    )
    if not changed:
        return False

    last_params["usercf_rows"] = u
    last_params["itemcf_rows"] = i
    last_params["svd_rows"] = s
    last_params["cf_rows"] = max(u, i, s)
    meta["last_train_params"] = last_params
    _save_training_meta(meta)
    return True


def cf_train_params_disk_tick():
    """
    用于协同页与磁盘最近一次 CF 训练元数据对齐：training_meta.json 或 cf_meta.json 变更时
    应刷新界面上的三路训练条数，避免 Streamlit session_state 长期残留旧值。
    """
    t = 0.0
    for p in (_training_meta_path(), CF_META_PATH):
        try:
            t += os.path.getmtime(p)
        except OSError:
            pass
    return t


def maybe_retrain_models_on_user_data_change(
    force=False,
    train_rows=None,
    train_rows_map=None,
    ncf_epochs=None,
    ncf_train_config=None,
    scopes=None,
):
    """
    若 users/history 发生变化，则重训并覆盖 CF/SVD/内容/NCF 模型。
    返回: (retrained: bool, message: str)
    """
    cur_sig = _current_user_data_signature()
    meta = _load_training_meta()
    last_sig = meta.get("user_data_signature")
    if (not force) and _normalize_sig(last_sig) == _normalize_sig(cur_sig):
        return False, "用户数据未变化，复用现有模型。"

    from src import recommend_utils as ru
    import src.deep_learning_recommend as dlr
    from src.deep_learning_recommend import train_ncf_model

    # 需要重训的模块范围：cf(整体) / usercf / itemcf / svd / content / ncf
    scope_set = None
    if scopes is None:
        scope_set = {"cf", "content", "ncf"}
    else:
        try:
            scope_set = {str(x).strip().lower() for x in (scopes or [])}
        except Exception:
            scope_set = {"cf", "content", "ncf"}
        scope_set = {x for x in scope_set if x in {"cf", "usercf", "itemcf", "svd", "content", "ncf"}}
        if not scope_set:
            scope_set = {"cf", "content", "ncf"}
    cf_sub_scope = {"usercf", "itemcf", "svd"}
    cf_scope_set = set(cf_sub_scope) if "cf" in scope_set else {x for x in scope_set if x in cf_sub_scope}

    # 可选：临时覆盖本次重训的数据量，避免固定上限导致内存压力过大
    row_override = None
    if train_rows is not None:
        try:
            row_override = int(train_rows)
        except Exception:
            row_override = None
    if row_override is not None:
        row_override = max(1000, row_override)

    row_overrides = {}
    if isinstance(train_rows_map, dict):
        for k in ("cf", "content", "ncf", "usercf", "itemcf", "svd"):
            v = train_rows_map.get(k)
            if v is None:
                continue
            try:
                row_overrides[k] = max(1000, int(v))
            except Exception:
                pass
    if row_override is not None:
        # 兼容旧参数：统一覆盖三种模型
        row_overrides = {"cf": row_override, "content": row_override, "ncf": row_override}

    old_cf_sample = getattr(ru, "CF_SAMPLE_SIZE", None)
    old_content_sample = getattr(ru, "CONTENT_SAMPLE_SIZE", None)
    old_ncf_sample = getattr(dlr, "NCF_SAMPLE_SIZE", None)
    # NCF epoch 可选覆盖
    epochs_override = None
    if ncf_epochs is not None:
        try:
            epochs_override = int(ncf_epochs)
        except Exception:
            epochs_override = None
    if epochs_override is not None:
        epochs_override = max(1, epochs_override)
    ncf_cfg = {}
    if isinstance(ncf_train_config, dict):
        ncf_cfg = dict(ncf_train_config)
    lr_override = ncf_cfg.get("lr", None)
    emb_dim_override = ncf_cfg.get("emb_dim", None)
    neg_k_override = ncf_cfg.get("neg_k", None)
    pop_ratio_override = ncf_cfg.get("popular_negative_ratio", None)
    min_epochs_override = ncf_cfg.get("min_epochs", None)
    patience_override = ncf_cfg.get("early_stop_patience", None)
    batch_size_override = ncf_cfg.get("batch_size", None)
    try:
        lr_override = float(lr_override) if lr_override is not None else 0.0006
    except Exception:
        lr_override = 0.0006
    try:
        emb_dim_override = max(8, int(emb_dim_override)) if emb_dim_override is not None else 64
    except Exception:
        emb_dim_override = 64
    try:
        neg_k_override = max(1, int(neg_k_override)) if neg_k_override is not None else 3
    except Exception:
        neg_k_override = 3
    try:
        pop_ratio_override = float(pop_ratio_override) if pop_ratio_override is not None else 0.67
    except Exception:
        pop_ratio_override = 0.67
    pop_ratio_override = min(1.0, max(0.0, pop_ratio_override))
    try:
        min_epochs_override = max(1, int(min_epochs_override)) if min_epochs_override is not None else 5
    except Exception:
        min_epochs_override = 5
    try:
        patience_override = max(1, int(patience_override)) if patience_override is not None else 4
    except Exception:
        patience_override = 4
    try:
        batch_size_override = max(64, int(batch_size_override)) if batch_size_override is not None else 1024
    except Exception:
        batch_size_override = 1024

    # 清理缓存与旧模型文件：按 scope 独立执行
    try:
        if cf_scope_set:
            ru._get_cf_models.clear()
        if "content" in scope_set:
            ru._get_feature_encoder.clear()
        if "ncf" in scope_set:
            ru._get_ncf_runtime.clear()
    except Exception:
        pass

    if "cf" in scope_set:
        try:
            for p in [CF_MODEL_BUNDLE_PATH, CF_USERCF_MODEL_PATH, CF_ITEMCF_MODEL_PATH, CF_SVD_MODEL_PATH, CF_META_PATH]:
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass
    if "content" in scope_set:
        try:
            if os.path.exists(CONTENT_MODEL_BUNDLE_PATH):
                os.remove(CONTENT_MODEL_BUNDLE_PATH)
        except Exception:
            pass
    if "ncf" in scope_set:
        try:
            if os.path.exists(NCF_MODEL_PATH):
                os.remove(NCF_MODEL_PATH)
        except Exception:
            pass

    try:
        if "cf" in row_overrides:
            ru.CF_SAMPLE_SIZE = row_overrides["cf"]
            ru.CF_USERCF_SAMPLE_SIZE = row_overrides["cf"]
            ru.CF_ITEMCF_SAMPLE_SIZE = row_overrides["cf"]
            ru.CF_SVD_SAMPLE_SIZE = row_overrides["cf"]
        if "usercf" in row_overrides:
            ru.CF_USERCF_SAMPLE_SIZE = row_overrides["usercf"]
        if "itemcf" in row_overrides:
            ru.CF_ITEMCF_SAMPLE_SIZE = row_overrides["itemcf"]
        if "svd" in row_overrides:
            ru.CF_SVD_SAMPLE_SIZE = row_overrides["svd"]
        if "content" in row_overrides:
            ru.CONTENT_SAMPLE_SIZE = row_overrides["content"]
        if "ncf" in row_overrides:
            dlr.NCF_SAMPLE_SIZE = row_overrides["ncf"]

        # 1) 协同过滤（按子模型粒度重训并写入持久化文件）
        if cf_scope_set:
            ru.retrain_cf_models(sorted(list(cf_scope_set)))
            try:
                ru._get_cf_models.clear()
            except Exception:
                pass

        # 2) 内容编码器（会自动写入持久化文件）
        if "content" in scope_set:
            ru._get_feature_encoder()

        # 3) NCF 主模型（覆盖保存）
        if "ncf" in scope_set:
            train_ncf_model(
                n_epochs=(epochs_override or 8),
                batch_size=batch_size_override,
                lr=lr_override,
                emb_dim=emb_dim_override,
                neg_k=neg_k_override,
                popular_negative_ratio=pop_ratio_override,
                min_epochs=min_epochs_override,
                early_stop_patience=patience_override,
            )
            try:
                ru._get_ncf_runtime.clear()
            except Exception:
                pass
    finally:
        if old_cf_sample is not None:
            ru.CF_SAMPLE_SIZE = old_cf_sample
            ru.CF_USERCF_SAMPLE_SIZE = old_cf_sample
            ru.CF_ITEMCF_SAMPLE_SIZE = old_cf_sample
            ru.CF_SVD_SAMPLE_SIZE = old_cf_sample
        if old_content_sample is not None:
            ru.CONTENT_SAMPLE_SIZE = old_content_sample
        if old_ncf_sample is not None:
            dlr.NCF_SAMPLE_SIZE = old_ncf_sample

    meta["user_data_signature"] = cur_sig
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["last_retrain_time"] = now_s
    # 记录分模块重训时间（工程上更直观）
    if cf_scope_set:
        meta["last_retrain_time_cf"] = now_s
    if "content" in scope_set:
        meta["last_retrain_time_content"] = now_s
    if "ncf" in scope_set:
        meta["last_retrain_time_ncf"] = now_s

    # 记录每个模型最近一次“页面重训”参数，供算法对比页复用
    last_params = meta.get("last_train_params", {})
    if not isinstance(last_params, dict):
        last_params = {}
    if cf_scope_set:
        old_u = int(last_params.get("usercf_rows", old_cf_sample or CF_SAMPLE_SIZE))
        old_i = int(last_params.get("itemcf_rows", old_cf_sample or CF_SAMPLE_SIZE))
        old_s = int(last_params.get("svd_rows", old_cf_sample or CF_SAMPLE_SIZE))
        base_cf = int(row_overrides["cf"]) if "cf" in row_overrides else max(old_u, old_i, old_s)
        if "usercf" in cf_scope_set or "cf" in scope_set:
            old_u = int(row_overrides.get("usercf", base_cf))
        if "itemcf" in cf_scope_set or "cf" in scope_set:
            old_i = int(row_overrides.get("itemcf", base_cf))
        if "svd" in cf_scope_set or "cf" in scope_set:
            old_s = int(row_overrides.get("svd", base_cf))
        last_params["usercf_rows"] = old_u
        last_params["itemcf_rows"] = old_i
        last_params["svd_rows"] = old_s
        last_params["cf_rows"] = max(old_u, old_i, old_s)
    if "content" in scope_set:
        last_params["content_rows"] = int(row_overrides.get("content", old_content_sample or CONTENT_SAMPLE_SIZE))
    if "ncf" in scope_set:
        last_params["ncf_rows"] = int(row_overrides.get("ncf", old_ncf_sample or NCF_SAMPLE_SIZE))
        last_params["ncf_epochs"] = int(epochs_override or 8)
        last_params["ncf_lr"] = float(lr_override)
        last_params["ncf_emb_dim"] = int(emb_dim_override)
        last_params["ncf_neg_k"] = int(neg_k_override)
        last_params["ncf_popular_negative_ratio"] = float(pop_ratio_override)
        last_params["ncf_min_epochs"] = int(min_epochs_override)
        last_params["ncf_patience"] = int(patience_override)
        last_params["ncf_batch_size"] = int(batch_size_override)
    meta["last_train_params"] = last_params
    _save_training_meta(meta)
    row_text = ""
    if row_overrides:
        bits = []
        if "cf" in row_overrides:
            bits.append(f"CF(统一条数)={row_overrides['cf']}")
        if "usercf" in row_overrides:
            bits.append(f"UserCF={row_overrides['usercf']}")
        if "itemcf" in row_overrides:
            bits.append(f"ItemCF={row_overrides['itemcf']}")
        if "svd" in row_overrides:
            bits.append(f"SVD={row_overrides['svd']}")
        if "content" in row_overrides:
            bits.append(f"Content={row_overrides['content']}")
        if "ncf" in row_overrides:
            bits.append(f"NCF={row_overrides['ncf']}")
        row_text = ("本次训练条数：" + "，".join(bits) + "。") if bits else ""
    if epochs_override is not None:
        row_text = (row_text + f" NCF_Epoch={epochs_override}。").strip()
    if "ncf" in scope_set:
        row_text = (
            row_text
            + f" NCF(lr={lr_override}, emb_dim={emb_dim_override}, neg_k={neg_k_override}, "
              f"popular_ratio={pop_ratio_override}, min_epochs={min_epochs_override}, patience={patience_override}, "
              f"batch={batch_size_override})。"
        ).strip()

    scope_text = "、".join(sorted(scope_set))
    scope_text = f"重训范围：{scope_text}。"
    row_text = (scope_text + " " + row_text).strip()
    if force:
        return True, f"已手动触发重训并覆盖模型（{meta['last_retrain_time']}）。{row_text}"
    return True, f"检测到用户数据变化，已完成重训并覆盖模型（{meta['last_retrain_time']}）。{row_text}"


def get_user_data_change_status():
    """返回用户数据是否有变化，供页面标记显示。"""
    cur_sig = _current_user_data_signature()
    meta = _load_training_meta()
    last_sig = meta.get("user_data_signature")
    changed = (_normalize_sig(last_sig) != _normalize_sig(cur_sig))
    last_time = meta.get("last_retrain_time", "未训练")
    if last_sig is None:
        msg = "尚未建立基线（首次推荐将重训）"
        changed = True
    elif changed:
        msg = "历史/用户数据有变化（需要重训）"
    else:
        msg = "历史/用户数据无变化（可复用模型）"
    return changed, msg, last_time


def save_training_snapshot_with_user_json(output_path=None):
    """
    将主训练表与 users.json + user_history.json 中的历史交互合并后导出。
    不修改原始 DATA_FILE，只输出快照文件，供后续训练/评测复用。
    Returns:
        (output_path, added_rows, total_rows)
    """
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if not {"user", "song", "play_count"}.issubset(set(df.columns)):
        raise ValueError("DATA_FILE 缺少必要字段 user/song/play_count")

    users = load_users()
    all_history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_history = json.load(f)

    # username -> user_id
    user_map = {}
    for uname, ud in users.items():
        try:
            uid = int(ud.get("user_id"))
            user_map[uname] = uid
        except Exception:
            continue

    new_pairs = []
    for uname, songs in all_history.items():
        if uname not in user_map:
            continue
        uid = user_map[uname]
        for sid in songs or []:
            try:
                new_pairs.append((uid, int(sid)))
            except Exception:
                continue

    if not new_pairs:
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(os.path.dirname(DATA_FILE), f"training_snapshot_{ts}.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path, 0, len(df)

    new_df = pd.DataFrame(new_pairs, columns=["user", "song"]).drop_duplicates()
    new_df["play_count"] = 1.0

    # 尽量补齐歌曲侧元信息列
    song_meta_cols = [c for c in df.columns if c not in {"user", "play_count"}]
    if "song" in song_meta_cols:
        song_meta = df[song_meta_cols].drop_duplicates(subset=["song"])
        new_df = new_df.merge(song_meta, on="song", how="left")
        if "play_count" in new_df.columns and "play_count_x" in new_df.columns:
            new_df["play_count"] = new_df["play_count_x"]
            new_df = new_df.drop(columns=[c for c in ["play_count_x", "play_count_y"] if c in new_df.columns])

    before = len(df)
    # 保证 new_df 与原表列对齐
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    new_df = new_df[df.columns]
    merged = pd.concat([df, new_df], ignore_index=True)
    # 为避免字段顺序被打乱，按原列重排并对 user-song 去重（保留较大偏好）
    for col in df.columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[df.columns]
    if {"user", "song", "play_count"}.issubset(set(merged.columns)):
        merged = merged.sort_values("play_count", ascending=False).drop_duplicates(subset=["user", "song"], keep="first")

    added = int(len(merged) - before)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(DATA_FILE), f"training_snapshot_{ts}.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, added, len(merged)
