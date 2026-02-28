# 融合推荐页面
import streamlit as st
from src.recommend_utils import hybrid_recommend, get_max_user_id_for_hybrid
from app.utils.ui_components import render_recommendation_results
from app.utils.helpers import get_max_user_id
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND


def render():
    # 渲染融合推荐页面
    st.title("融合推荐（多算法加权融合）")
    
    # 获取当前用户信息
    user_id = st.session_state.get('user_id', None)
    user_history = st.session_state.get('user_history', [])
    username = st.session_state.get('username', '')
    
    if user_id is None:
        st.warning("⚠️ 请先登录")
        return
    
    # 融合推荐说明：根据用户类型自动调整算法权重
    
    st.write("---")
    
    # 显示用户信息
    st.write(f"**当前用户：** {username}（用户ID: {user_id}）")
    st.write(f"**历史记录数量：** {len(user_history)} 条")
    
    # 权重设置
    st.markdown("### 权重设置（可选）")
    use_custom_weights = st.checkbox("使用自定义权重", value=False)
    
    if use_custom_weights:
        col1, col2, col3 = st.columns(3)
        with col1:
            weight_cf = st.slider("协同过滤权重", 0.0, 1.0, 0.4, 0.1)
        with col2:
            try:
                max_user_id = get_max_user_id()
            except:
                max_user_id = get_max_user_id_for_hybrid()
            can_use_ncf = user_id is not None and user_id <= max_user_id
            can_use_ncf_with_history = can_use_ncf or (user_id is not None and len(user_history) > 0)
            
            if can_use_ncf_with_history:
                weight_ncf = st.slider("深度学习权重", 0.0, 1.0, 0.4, 0.1)
                if not can_use_ncf and len(user_history) > 0:
                    st.caption("将使用个性化训练模式")
            else:
                st.info("NCF需要历史记录或用户ID在训练范围内")
                weight_ncf = 0.0
        with col3:
            weight_content = st.slider("内容推荐权重", 0.0, 1.0, 0.2, 0.1)
        
        # 归一化权重
        total = weight_cf + weight_ncf + weight_content
        if total > 0:
            weight_cf /= total
            weight_ncf /= total
            weight_content /= total
        else:
            st.warning("权重总和不能为0")
            weight_cf, weight_ncf, weight_content = 0.4, 0.4, 0.2
        
        weights = {'cf': weight_cf, 'ncf': weight_ncf, 'content': weight_content}
        st.write(f"**当前权重：** CF={weight_cf:.2f}, NCF={weight_ncf:.2f}, 内容={weight_content:.2f}")
    else:
        weights = None
    
    # 推荐数量选择
    topk = st.slider(
        "推荐TopN",
        min_value=MIN_RECOMMEND,
        max_value=MAX_RECOMMEND,
        value=DEFAULT_RECOMMEND,
        step=1
    )
    
    # 获取融合推荐按钮
    if st.button("获取融合推荐", type="primary"):
        try:
            with st.spinner("正在融合多种算法生成推荐..."):
                result = hybrid_recommend(user_id, user_history, topk, weights)
                st.session_state['hybrid_recommend_result'] = result
            
            if result:
                st.success("融合推荐生成完成！")
            else:
                st.warning("未找到推荐结果")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # 显示推荐结果
    if 'hybrid_recommend_result' in st.session_state:
        result = st.session_state['hybrid_recommend_result']
        st.write("**融合推荐结果：**")
        render_recommendation_results(result, prefix='hybrid', show_listen_button=True)
