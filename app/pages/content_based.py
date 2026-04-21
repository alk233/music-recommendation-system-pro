# 内容推荐页面
import streamlit as st
from src.recommend_utils import content_based_recommend
from app.utils.ui_components import render_recommendation_results, render_history_section
from app.utils.helpers import extract_song_id, save_to_history
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND, MAX_HISTORY_SIZE, CONTENT_SAMPLE_SIZE
from app.utils.data_notes import render_page_metric_note
from app.utils.background_prefetch import cancel_idle_prefetch, get_prefetch_result
from app.utils.helpers import (
    get_user_data_change_status,
    maybe_retrain_models_on_user_data_change,
    get_data_total_rows,
    get_last_model_train_params,
)


def render():
    # 渲染内容推荐页面
    st.title("内容推荐（基于内容特征+用户历史）")
    render_page_metric_note("content")

    # 显示用户ID
    if 'username' in st.session_state:
        st.write(f"**当前用户ID：** {st.session_state['username']}")
    changed, change_msg, last_time = get_user_data_change_status()
    st.caption(f"数据变更标记：{change_msg}；上次重训：{last_time}")
    dynamic_max_rows = int(get_data_total_rows())
    # 重启后优先恢复最近一次内容模型重训条数
    last_params = get_last_model_train_params()
    st.session_state.setdefault("force_retrain_rows_content", int(last_params.get("content_rows", CONTENT_SAMPLE_SIZE)))

    c_train_rows, c_force_btn = st.columns([2, 1])
    with c_train_rows:
        train_rows_content = st.number_input(
            "内容模型重训数据条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_content", CONTENT_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_content",
            help="仅设置本页触发重训时的内容模型训练条数。",
        )
    with c_force_btn:
        if st.button("强制重新训练模型", key="force_retrain_content"):
            cancel_idle_prefetch()
            with st.spinner("正在强制重训模型并覆盖保存..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    force=True,
                    train_rows_map={"content": int(train_rows_content)},
                    scopes=["content"],
                )
            if retrained:
                st.success(retrain_msg)
            else:
                st.info(retrain_msg)
    
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
    user_history = st.session_state.get('user_history', [])
    history_signature = tuple(sorted(set(user_history)))
    cache = st.session_state.setdefault("content_result_cache", {})
    cache_key = f"{st.session_state.get('user_id', None)}|{topk}|{history_signature}"

    if st.button("获取内容推荐", type="primary"):
        user_history = st.session_state.get('user_history', [])
        
        if not user_history:
            st.warning("⚠️ 你还没有历史记录，请先在冷启动推荐页面勾选一些歌曲并保存到历史")
            return
        
        try:
            cancel_idle_prefetch()
            with st.spinner("正在根据你的历史记录生成个性化推荐..."):
                prefetched = None
                if topk == DEFAULT_RECOMMEND:
                    prefetched = get_prefetch_result(
                        "content",
                        st.session_state.get('username', ''),
                        st.session_state.get('user_id', None),
                        user_history,
                        topk=topk
                    )
                st.session_state['content_recommend_result'] = prefetched if prefetched else content_based_recommend(
                    user_history,
                    topk
                )
                cache[cache_key] = st.session_state['content_recommend_result']
            st.success("推荐生成完成！")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
    
    # 显示推荐结果
    if cache_key in cache:
        result = cache[cache_key]
        st.session_state['content_recommend_result'] = result
        st.write("**为你推荐的歌曲：**")
        
        # 渲染推荐结果（每个歌曲后面有"听歌"按钮）
        render_recommendation_results(result, prefix='content', show_listen_button=True)
