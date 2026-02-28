# 内容推荐页面
import streamlit as st
from src.recommend_utils import content_based_recommend
from app.utils.ui_components import render_recommendation_results, render_history_section
from app.utils.helpers import extract_song_id, save_to_history
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND, MAX_HISTORY_SIZE


def render():
    # 渲染内容推荐页面
    st.title("内容推荐（基于内容特征+用户历史）")
    
    # 显示用户ID
    if 'username' in st.session_state:
        st.write(f"**当前用户ID：** {st.session_state['username']}")
    
    # 显示历史记录
    render_history_section()
    
    # 推荐数量选择
    topk = st.slider(
        "推荐TopN",
        min_value=MIN_RECOMMEND,
        max_value=MAX_RECOMMEND,
        value=DEFAULT_RECOMMEND,
        step=1
    )
    
    # 获取内容推荐按钮
    if st.button("获取内容推荐", type="primary"):
        user_history = st.session_state.get('user_history', [])
        
        if not user_history:
            st.warning("⚠️ 你还没有历史记录，请先在冷启动推荐页面勾选一些歌曲并保存到历史")
            return
        
        try:
            with st.spinner("正在根据你的历史记录生成个性化推荐..."):
                st.session_state['content_recommend_result'] = content_based_recommend(
                    user_history,
                    topk
                )
            st.success("推荐生成完成！")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
    
    # 显示推荐结果
    if 'content_recommend_result' in st.session_state:
        result = st.session_state['content_recommend_result']
        st.write("**为你推荐的歌曲：**")
        
        # 渲染推荐结果（每个歌曲后面有"听歌"按钮）
        render_recommendation_results(result, prefix='content', show_listen_button=True)
