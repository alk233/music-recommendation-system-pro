# 协同过滤推荐页面
import streamlit as st
from src.recommend_utils import usercf_topn, itemcf_topn, svd_topn
from app.utils.ui_components import render_recommendation_results
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND


def render():
    # 渲染协同过滤推荐页面
    st.title("排行榜 - 协同过滤推荐")
    
    # 选择算法
    model_type = st.radio(
        "选择协同过滤模型",
        ["UserCF", "ItemCF", "SVD"],
        horizontal=True
    )
    
    # 选择推荐数量
    topk = st.slider(
        "推荐TopN",
        min_value=MIN_RECOMMEND,
        max_value=MAX_RECOMMEND,
        value=DEFAULT_RECOMMEND,
        step=1
    )
    
    # 获取推荐结果
    try:
        with st.spinner(f"正在使用{model_type}算法生成推荐..."):
            if model_type == "UserCF":
                result = usercf_topn(topk)
            elif model_type == "ItemCF":
                result = itemcf_topn(topk)
            else:
                result = svd_topn(topk)
        
        st.write(f"**{model_type} 推荐Top{topk}：**")
        render_recommendation_results(result, prefix='cf', show_listen_button=True)
        
    except Exception as e:
        st.error(f"推荐生成失败: {str(e)}")
        st.info("请确保数据文件存在且格式正确")
