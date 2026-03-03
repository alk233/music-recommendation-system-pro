# 音乐推荐系统主入口
import streamlit as st
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import PAGE_TITLE, PAGE_LAYOUT
from app.utils.ui_components import render_sidebar_history, render_user_login
from app.pages import cold_start, collaborative, deep_learning, content_based, hybrid, analysis

# 页面配置
st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

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
            # 清除会话状态
            for key in ['username', 'user_id', 'user_history', 'history_loaded']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

page = st.sidebar.radio(
    "导航",
    ["首页（热门推荐）", "排行榜（协同过滤）", "深度学习推荐", "内容推荐", "融合推荐", "必吃榜"]
)

# 渲染侧边栏历史记录
render_sidebar_history()

# 用户登录
if not render_user_login():
    st.stop()

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
    elif page == "必吃榜":
        analysis.render()
except Exception as e:
    st.error(f"页面加载失败: {str(e)}")
    st.info("请检查数据文件和模型文件是否存在")
