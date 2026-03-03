# 工具函数模块
import os
import json
import requests
import base64
import hashlib
import hmac
import pandas as pd
import streamlit as st
from config import FONT_PATH, FONT_URL, DATA_FILE, HISTORY_FILE, MAX_HISTORY_SIZE, NCF_SAMPLE_SIZE, USERS_FILE


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
            usecols=['song', 'artist_name', 'title']
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
                return all_history.get(username, [])
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

def get_user_history_display():
    # 获取用户历史记录的显示文本列表
    if 'user_history' not in st.session_state or not st.session_state['user_history']:
        return []
    
    df_hist = load_song_info()
    if df_hist.empty:
        return [f"song_id={sid}" for sid in st.session_state['user_history']]
    
    result = []
    for sid in st.session_state['user_history']:
        if sid in df_hist.index:
            artist = df_hist.loc[sid, 'artist_name']
            title = df_hist.loc[sid, 'title']
            result.append(f"{artist} - {title} (song_id={sid})")
        else:
            result.append(f"song_id={sid}")
    return result
