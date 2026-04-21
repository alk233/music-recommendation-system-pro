# 深度学习推荐页面
import streamlit as st
from src.recommend_utils import ncf_recommend
from app.utils.ui_components import render_recommendation_results
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND, NCF_SAMPLE_SIZE
from app.utils.data_notes import render_page_metric_note
from app.utils.background_prefetch import cancel_idle_prefetch, get_prefetch_result
from app.utils.helpers import (
    maybe_retrain_models_on_user_data_change,
    get_user_data_change_status,
    get_data_total_rows,
    get_last_model_train_params,
)


def render():
    # 渲染深度学习推荐页面
    from app.utils.helpers import get_max_user_id
    
    st.title("深度学习推荐（NCF）")
    render_page_metric_note("ncf")

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
    changed, change_msg, last_time = get_user_data_change_status()
    st.caption(f"数据变更标记：{change_msg}；上次重训：{last_time}")
    dynamic_max_rows = int(get_data_total_rows())
    # 重启后优先恢复最近一次 NCF 重训参数
    last_params = get_last_model_train_params()
    st.session_state.setdefault("force_retrain_rows_ncf", int(last_params.get("ncf_rows", NCF_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_ncf_epochs", int(last_params.get("ncf_epochs", 8)))
    st.session_state.setdefault("force_retrain_ncf_lr", float(last_params.get("ncf_lr", 0.0006)))
    st.session_state.setdefault("force_retrain_ncf_emb_dim", int(last_params.get("ncf_emb_dim", 64)))
    st.session_state.setdefault("force_retrain_ncf_neg_k", int(last_params.get("ncf_neg_k", 3)))
    st.session_state.setdefault("force_retrain_ncf_pop_ratio", float(last_params.get("ncf_popular_negative_ratio", 0.67)))
    st.session_state.setdefault("force_retrain_ncf_min_epochs", int(last_params.get("ncf_min_epochs", 5)))
    st.session_state.setdefault("force_retrain_ncf_patience", int(last_params.get("ncf_patience", 4)))
    st.session_state.setdefault("force_retrain_ncf_batch_size", int(last_params.get("ncf_batch_size", 1024)))

    c_train_rows, c_epochs, c_force_btn = st.columns([2, 1, 1])
    with c_train_rows:
        train_rows_ncf = st.number_input(
            "NCF重训数据条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_ncf", NCF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_ncf",
            help="仅设置本页触发重训时的 NCF 训练条数。",
        )
    with c_epochs:
        ncf_epochs = st.number_input(
            "NCF Epoch",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("force_retrain_ncf_epochs", 8)),
            step=1,
            key="force_retrain_ncf_epochs",
            help="本页强制重训时 NCF 训练轮数（epoch）。",
        )
    with st.expander("NCF高级参数（训练）", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            ncf_lr = st.number_input(
                "学习率 lr",
                min_value=0.00001,
                max_value=0.01,
                value=float(st.session_state.get("force_retrain_ncf_lr", 0.0006)),
                step=0.0001,
                format="%.5f",
                key="force_retrain_ncf_lr",
            )
        with p2:
            ncf_emb_dim = st.number_input(
                "Embedding维度 emb_dim",
                min_value=8,
                max_value=256,
                value=int(st.session_state.get("force_retrain_ncf_emb_dim", 64)),
                step=8,
                key="force_retrain_ncf_emb_dim",
            )
        with p3:
            ncf_neg_k = st.number_input(
                "每正样本负采样数 neg_k",
                min_value=1,
                max_value=10,
                value=int(st.session_state.get("force_retrain_ncf_neg_k", 3)),
                step=1,
                key="force_retrain_ncf_neg_k",
            )
        p4, p5, p6, p7 = st.columns(4)
        with p4:
            ncf_pop_ratio = st.slider(
                "热门负样本占比",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("force_retrain_ncf_pop_ratio", 0.67)),
                step=0.01,
                key="force_retrain_ncf_pop_ratio",
            )
        with p5:
            ncf_min_epochs = st.number_input(
                "最少训练轮数 min_epochs",
                min_value=1,
                max_value=30,
                value=int(st.session_state.get("force_retrain_ncf_min_epochs", 5)),
                step=1,
                key="force_retrain_ncf_min_epochs",
            )
        with p6:
            ncf_patience = st.number_input(
                "早停耐心 patience",
                min_value=1,
                max_value=15,
                value=int(st.session_state.get("force_retrain_ncf_patience", 4)),
                step=1,
                key="force_retrain_ncf_patience",
            )
        with p7:
            ncf_batch_size = st.number_input(
                "批大小 batch_size",
                min_value=64,
                max_value=8192,
                value=int(st.session_state.get("force_retrain_ncf_batch_size", 1024)),
                step=64,
                key="force_retrain_ncf_batch_size",
            )
        st.caption("建议：emb_dim 64→96，neg_k 3→4，lr 6e-4→5e-4，epoch 8→10，patience 保持 4。")
    with c_force_btn:
        if st.button("强制重新训练模型", key="force_retrain_ncf"):
            cancel_idle_prefetch()
            with st.spinner("正在强制重训模型并覆盖保存..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    force=True,
                    train_rows_map={"ncf": int(train_rows_ncf)},
                    ncf_epochs=int(ncf_epochs),
                    ncf_train_config={
                        "lr": float(ncf_lr),
                        "emb_dim": int(ncf_emb_dim),
                        "neg_k": int(ncf_neg_k),
                        "popular_negative_ratio": float(ncf_pop_ratio),
                        "min_epochs": int(ncf_min_epochs),
                        "early_stop_patience": int(ncf_patience),
                        "batch_size": int(ncf_batch_size),
                    },
                    scopes=["ncf"],
                )
            if retrained:
                st.success(retrain_msg)
            else:
                st.info(retrain_msg)
    
    # 检查用户ID是否在训练数据范围内
    user_history = st.session_state.get('user_history', [])
    
    if not can_use_ncf:
        if user_history:
            st.info(f"""
            ℹ️ **个性化训练模式：**
            
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
    history_signature = tuple(sorted(set(user_history)))
    mode_hint = "in_train" if can_use_ncf else "new_user"
    cache = st.session_state.setdefault("ncf_result_cache", {})
    cache_key = f"{user_id}|{topk}|{mode_hint}|{history_signature}"

    if st.button("获取推荐", type="primary"):
        try:
            cancel_idle_prefetch()
            with st.spinner("检查用户数据变化并自动重训模型..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    train_rows_map={"ncf": int(train_rows_ncf)},
                    ncf_epochs=int(ncf_epochs),
                    ncf_train_config={
                        "lr": float(ncf_lr),
                        "emb_dim": int(ncf_emb_dim),
                        "neg_k": int(ncf_neg_k),
                        "popular_negative_ratio": float(ncf_pop_ratio),
                        "min_epochs": int(ncf_min_epochs),
                        "early_stop_patience": int(ncf_patience),
                        "batch_size": int(ncf_batch_size),
                    },
                    scopes=["ncf"],
                )
            if retrained:
                st.success(retrain_msg)
            user_history = st.session_state.get('user_history', [])
            prefetched = None
            if topk == DEFAULT_RECOMMEND and not retrained:
                prefetched = get_prefetch_result("ncf", username, user_id, user_history, topk=topk)

            if prefetched:
                result = prefetched
                mode = "personalized" if user_history else "pretrained"
            elif can_use_ncf:
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
            cache[cache_key] = result
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
    if cache_key in cache:
        result = cache[cache_key]
        st.session_state['ncf_recommend_result'] = result
        if result:
            st.write("**为你推荐的歌曲：**")
            render_recommendation_results(result, prefix='ncf', show_listen_button=True)
        else:
            st.warning("未找到推荐结果，请尝试积累更多历史记录")
