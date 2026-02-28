# UI组件模块
import streamlit as st
from app.utils.helpers import get_user_history_display, load_song_info


def render_sidebar_history():
    # 渲染侧边栏历史记录
    from app.utils.helpers import extract_song_id, remove_from_history
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("你的历史歌曲（最近10条）")
    
    history_display = get_user_history_display()
    if history_display:
        user_history = st.session_state.get('user_history', [])
        for idx, item in enumerate(history_display):
            col1, col2 = st.sidebar.columns([5, 1])
            with col1:
                st.sidebar.markdown(f"<div style='font-size: 0.9em;'>{item}</div>", unsafe_allow_html=True)
            with col2:
                if idx < len(user_history):
                    song_id = user_history[idx]
                    delete_key = f'sidebar_delete_{song_id}_{idx}'
                    if st.sidebar.button("🗑️", key=delete_key, help="删除", use_container_width=True):
                        if remove_from_history(song_id):
                            st.rerun()
    else:
        st.sidebar.write("暂无历史记录")


def render_user_login():
    # 渲染用户登录/注册组件
    from app.utils.helpers import (
        load_user_history_from_file, get_max_user_id,
        register_user, verify_user, get_user_id_by_username
    )
    
    if 'username' not in st.session_state:
        st.session_state['username'] = ''
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'user_history' not in st.session_state:
        st.session_state['user_history'] = []
    if 'history_loaded' not in st.session_state:
        st.session_state['history_loaded'] = False
    
    # 获取最大用户ID
    max_user_id = get_max_user_id()
    
    # 如果已登录但历史记录未加载，则加载历史记录
    if st.session_state['username'] and not st.session_state['history_loaded']:
        history = load_user_history_from_file(st.session_state['username'])
        st.session_state['user_history'] = history
        st.session_state['history_loaded'] = True
    
    if not st.session_state['username']:
        st.info("👤 请先登录或注册")
        
        # 使用tabs切换登录和注册
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            st.markdown("### 登录")
            # 登录说明
            
            login_username = st.text_input("用户名", key="login_username")
            login_password = st.text_input("密码", type="password", key="login_password")
            login_clicked = st.button("登录", type="primary", key="login_btn")
            
            if login_clicked:
                success, user_id, message = verify_user(login_username, login_password)
                if success:
                    st.session_state['username'] = login_username
                    st.session_state['user_id'] = user_id
                    # 加载历史记录
                    history = load_user_history_from_file(login_username)
                    st.session_state['user_history'] = history
                    st.session_state['history_loaded'] = True
                    if history:
                        st.success(f"欢迎回来，{login_username}！已加载 {len(history)} 条历史记录")
                    else:
                        st.success(f"欢迎，{login_username}！")
                    st.rerun()
                else:
                    st.error(message)
        
        with tab2:
            st.markdown("### 注册新用户")
            # 注册说明
            
            reg_username = st.text_input("用户名", key="reg_username")
            reg_password = st.text_input("密码", type="password", key="reg_password")
            reg_password_confirm = st.text_input("确认密码", type="password", key="reg_password_confirm")
            register_clicked = st.button("注册", type="primary", key="register_btn")
            
            if register_clicked:
                if reg_password != reg_password_confirm:
                    st.error("两次输入的密码不一致")
                elif not reg_username or not reg_password:
                    st.error("用户名和密码不能为空")
                else:
                    success, message = register_user(reg_username, reg_password)
                    if success:
                        st.success(message)
                        # 自动登录
                        user_id = get_user_id_by_username(reg_username)
                        st.session_state['username'] = reg_username
                        st.session_state['user_id'] = user_id
                        st.session_state['user_history'] = []
                        st.session_state['history_loaded'] = True
                        st.rerun()
                    else:
                        st.error(message)
        
        if not st.session_state['username']:
            st.stop()
        return False
    return True


def render_recommendation_results(results, prefix='', show_listen_button=True):
    # 渲染推荐结果
    from app.utils.helpers import extract_song_id, save_to_history
    from config import MAX_HISTORY_SIZE
    
    # 渲染推荐结果和按钮
    for idx, row in enumerate(results):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(row)
        with col2:
            if show_listen_button:
                button_key = f'{prefix}_listen_{idx}'
                if st.button("🎵 听歌", key=button_key, use_container_width=True):
                    song_id = extract_song_id(row)
                    if song_id:
                        # 保存到历史
                        save_to_history([song_id], MAX_HISTORY_SIZE)
                        st.session_state[f'{prefix}_just_added'] = True
                        st.session_state[f'{prefix}_added_song'] = row[:60]
    
    # 检查是否有歌曲刚被添加
    if st.session_state.get(f'{prefix}_just_added', False):
        added_song = st.session_state.get(f'{prefix}_added_song', '')
        st.success(f"✅ 已添加到历史：{added_song}...")
        st.session_state[f'{prefix}_just_added'] = False
        st.rerun()


def render_history_section():
    # 渲染页面中的历史记录部分
    from app.utils.helpers import extract_song_id, remove_from_history
    
    st.write("你的最近10条历史歌曲：")
    history_display = get_user_history_display()
    if history_display:
        user_history = st.session_state.get('user_history', [])
        for idx, item in enumerate(history_display):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(item)
            with col2:
                if idx < len(user_history):
                    song_id = user_history[idx]
                    delete_key = f'content_delete_{song_id}_{idx}'
                    if st.button("🗑️ 删除", key=delete_key, help="删除这条记录", use_container_width=True):
                        if remove_from_history(song_id):
                            st.success("已删除")
                            st.rerun()
    else:
        st.write("暂无历史记录")
