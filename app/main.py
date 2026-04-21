# 音乐推荐系统主入口
import streamlit as st
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import PAGE_TITLE, PAGE_LAYOUT, DATA_FILE, BASE_DIR as CONFIG_BASE_DIR, DATA_DIR
from app.utils.ui_components import render_sidebar_history, render_user_login
from app.utils.background_prefetch import start_idle_prefetch, cancel_idle_prefetch
from app.pages import (
    cold_start,
    collaborative,
    deep_learning,
    content_based,
    hybrid,
    analysis,
    algorithm_compare,
)

# 页面配置
st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

if not os.path.isfile(DATA_FILE):
    st.error("找不到主数据文件 `final_merged_encoded_usernorm.csv`。")
    st.markdown(
        "请将数据处理生成的该文件放到以下**任一**位置：\n\n"
        f"1. **项目根目录**（推荐）：`{os.path.join(CONFIG_BASE_DIR, 'final_merged_encoded_usernorm.csv')}`\n\n"
        f"2. **data 子目录**：`{os.path.join(DATA_DIR, 'final_merged_encoded_usernorm.csv')}`\n\n"
        "若尚未生成，请在项目根目录执行：`python data_processing/run_all.py`"
    )
    st.stop()

# 隐藏默认导航菜单
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebarNav"] {visibility: hidden;}
[data-testid="stSidebarNav"] ul {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("🎵 音乐推荐系统")

# 显示用户信息和登出按钮
if 'username' in st.session_state and st.session_state['username']:
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.sidebar.markdown(f"**当前用户：** {st.session_state['username']}")
        if 'user_id' in st.session_state and st.session_state['user_id'] is not None:
            st.sidebar.markdown(f"**用户ID：** {st.session_state['user_id']}")
    with col2:
        if st.sidebar.button("登出", key="logout_btn", use_container_width=True):
            cancel_idle_prefetch()
            # 清除会话状态
            for key in ['username', 'user_id', 'user_history', 'history_loaded', 'history_view_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

page = st.sidebar.radio(
    "导航",
    [
        "首页（热门推荐）",
        "排行榜（协同过滤）",
        "深度学习推荐",
        "内容推荐",
        "融合推荐",
        "算法对比评测",
        "个人中心",
    ],
)

# 渲染侧边栏历史记录
render_sidebar_history()

# 用户登录
if not render_user_login():
    st.stop()

last_page = st.session_state.get("_last_page")
if last_page is not None and last_page != page:
    cancel_idle_prefetch()
st.session_state["_last_page"] = page

start_idle_prefetch(
    st.session_state.get("username"),
    st.session_state.get("user_id"),
    st.session_state.get("user_history", []),
)

# 根据选择渲染对应页面
try:
    if page == "首页（热门推荐）":
        cold_start.render()
    elif page == "排行榜（协同过滤）":
        collaborative.render()
    elif page == "深度学习推荐":
        deep_learning.render()
    elif page == "内容推荐":
        content_based.render()
    elif page == "融合推荐":
        hybrid.render()
    elif page == "算法对比评测":
        algorithm_compare.render()
    elif page == "个人中心":
        analysis.render()
except Exception as e:
    st.error(f"页面加载失败: {str(e)}")
    st.info("请检查数据文件和模型文件是否存在")
