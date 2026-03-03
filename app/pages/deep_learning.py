# 深度学习推荐页面
import streamlit as st
from src.recommend_utils import ncf_recommend
from app.utils.ui_components import render_recommendation_results
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND


def render():
    # 渲染深度学习推荐页面
    from app.utils.helpers import get_max_user_id
    
    st.title("深度学习推荐（NCF）")
    
    # 获取当前登录的用户ID
    user_id = st.session_state.get('user_id', None)
    username = st.session_state.get('username', '')
    
    if user_id is None:
        st.warning("⚠️ 请先登录")
        return
    
    max_user_id = get_max_user_id()
    can_use_ncf = user_id is not None and user_id <= max_user_id
    
    st.markdown(f"""
    **当前用户：{username}（用户ID: {user_id}）**
    """)
    
    # 检查用户ID是否在训练数据范围内
    user_history = st.session_state.get('user_history', [])
    
    if not can_use_ncf:
        if user_history:
            st.info(f"""
            ℹ️ **个性化训练模式：**
            
            - 你的用户ID（{user_id}）超出了训练数据范围（0-{max_user_id}）
            - 系统将使用你的历史记录（{len(user_history)}条）进行**实时个性化训练**
            - 训练过程会结合你的历史记录和全局数据，为你生成个性化推荐
            - 训练时间约几秒钟，请耐心等待
            """)
        else:
            st.warning(f"""
            ⚠️ **注意：** 你的用户ID（{user_id}）超出了训练数据范围（0-{max_user_id}）
            
            - NCF模型只在训练数据范围内的用户ID（0-{max_user_id}）上训练
            - 你的用户ID（{user_id}）是注册时新分配的，不在训练数据中
            - **建议：** 先听一些歌曲积累历史记录，然后系统可以为你进行个性化训练
            - 或者使用其他推荐算法（内容推荐、协同过滤）来获取推荐结果
            """)
            st.stop()
    
    # 使用用户ID进行NCF推荐
    
    st.write("---")
    
    # 推荐数量
    topk = st.slider(
        "推荐TopN",
        min_value=MIN_RECOMMEND,
        max_value=MAX_RECOMMEND,
        value=DEFAULT_RECOMMEND,
        step=1
    )
    
    # 获取推荐按钮
    if st.button("获取推荐", type="primary"):
        try:
            user_history = st.session_state.get('user_history', [])
            
            if can_use_ncf:
                # 使用预训练模型
                with st.spinner(f"正在使用NCF模型为用户ID {username} 生成推荐..."):
                    result = ncf_recommend(user_id, topk, user_history)
                mode = "pretrained"
            else:
                # 使用个性化训练
                if user_history:
                    with st.spinner(f"正在基于你的历史记录进行个性化训练（{len(user_history)}条历史）..."):
                        result = ncf_recommend(user_id, topk, user_history)
                    mode = "personalized"
                else:
                    st.warning("请先积累一些历史记录（听一些歌曲）")
                    result = []
                    mode = "personalized"
            
            st.session_state['ncf_recommend_result'] = result
            st.session_state['ncf_recommend_meta'] = {
                'user_id': user_id,
                'topk': topk,
                'mode': mode
            }
            if mode == "personalized" and result:
                st.success("个性化训练完成！")
                
        except FileNotFoundError:
            st.error("模型文件不存在，请先训练模型并确保文件在 model/ncf_model.pth")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # 显示推荐结果（重跑后也保留）
    meta = st.session_state.get('ncf_recommend_meta', {})
    if (
        'ncf_recommend_result' in st.session_state
        and meta.get('user_id') == user_id
    ):
        result = st.session_state['ncf_recommend_result']
        if result:
            st.write("**为你推荐的歌曲：**")
            render_recommendation_results(result, prefix='ncf', show_listen_button=True)
        else:
            st.warning("未找到推荐结果，请尝试积累更多历史记录")
