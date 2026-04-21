# 融合推荐页面
import streamlit as st
from src.recommend_utils import (
    hybrid_recommend,
    get_max_user_id_for_hybrid,
    default_hybrid_weights_5way,
)
from app.utils.ui_components import render_recommendation_results
from app.utils.helpers import (
    get_max_user_id,
    get_user_data_change_status,
    maybe_retrain_models_on_user_data_change,
    get_data_total_rows,
    get_last_model_train_params,
)
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND, CF_SAMPLE_SIZE, CONTENT_SAMPLE_SIZE, NCF_SAMPLE_SIZE
from app.utils.data_notes import render_page_metric_note
from app.utils.background_prefetch import cancel_idle_prefetch, get_prefetch_result


def render():
    # 渲染融合推荐页面
    st.title("融合推荐（多算法加权融合）")
    render_page_metric_note("hybrid")
    st.caption(
        "线上一键融合同时纳入 UserCF、ItemCF、SVD、NCF 与内容推荐五路分数，在候选并集上分别校准后加权；"
        "与算法对比页评测使用同一套加权思路。详情见「指标说明」。"
    )

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
    dynamic_max_rows = int(get_data_total_rows())
    changed, change_msg, last_time = get_user_data_change_status()
    st.caption(f"数据变更标记：{change_msg}；上次重训：{last_time}")
    st.caption(
        f"各模型「训练条数」动态上限（主数据文件总行数，缓存约 60 秒）：{dynamic_max_rows:,} 行"
    )
    # 重启后优先恢复最近一次参数；兼容旧版单一的 CF条数
    last_params = get_last_model_train_params()
    if (
        "force_retrain_rows_usercf_hybrid" not in st.session_state
        and "force_retrain_rows_cf_hybrid" in st.session_state
    ):
        legacy = int(st.session_state["force_retrain_rows_cf_hybrid"])
        st.session_state["force_retrain_rows_usercf_hybrid"] = legacy
        st.session_state["force_retrain_rows_itemcf_hybrid"] = legacy
        st.session_state["force_retrain_rows_svd_hybrid"] = legacy
    st.session_state.setdefault("force_retrain_rows_usercf_hybrid", int(last_params.get("usercf_rows", CF_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_rows_itemcf_hybrid", int(last_params.get("itemcf_rows", CF_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_rows_svd_hybrid", int(last_params.get("svd_rows", CF_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_rows_content_hybrid", int(last_params.get("content_rows", CONTENT_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_rows_ncf_hybrid", int(last_params.get("ncf_rows", NCF_SAMPLE_SIZE)))
    st.session_state.setdefault("force_retrain_ncf_epochs_hybrid", int(last_params.get("ncf_epochs", 8)))
    st.session_state.setdefault("force_retrain_ncf_lr_hybrid", float(last_params.get("ncf_lr", 0.0006)))
    st.session_state.setdefault("force_retrain_ncf_emb_dim_hybrid", int(last_params.get("ncf_emb_dim", 64)))
    st.session_state.setdefault("force_retrain_ncf_neg_k_hybrid", int(last_params.get("ncf_neg_k", 3)))
    st.session_state.setdefault("force_retrain_ncf_pop_ratio_hybrid", float(last_params.get("ncf_popular_negative_ratio", 0.67)))
    st.session_state.setdefault("force_retrain_ncf_min_epochs_hybrid", int(last_params.get("ncf_min_epochs", 5)))
    st.session_state.setdefault("force_retrain_ncf_patience_hybrid", int(last_params.get("ncf_patience", 4)))
    st.session_state.setdefault("force_retrain_ncf_batch_size_hybrid", int(last_params.get("ncf_batch_size", 1024)))

    cu, ci, cs = st.columns(3)
    with cu:
        train_rows_usercf = st.number_input(
            "UserCF训练条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_usercf_hybrid", CF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_usercf_hybrid",
            help="融合页重训时 UserCF 读取的交互行数（与 ItemCF/SVD 可分别设置）。",
        )
    with ci:
        train_rows_itemcf = st.number_input(
            "ItemCF训练条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_itemcf_hybrid", CF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_itemcf_hybrid",
            help="融合页重训时 ItemCF 读取的交互行数。",
        )
    with cs:
        train_rows_svd = st.number_input(
            "SVD训练条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_svd_hybrid", CF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_svd_hybrid",
            help="融合页重训时 SVD 读取的交互行数。",
        )

    c_content, c_ncf, c_epochs, c_force_btn = st.columns([1, 1, 1, 1])
    with c_content:
        train_rows_content = st.number_input(
            "内容条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_content_hybrid", CF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_content_hybrid",
            help="融合页重训时内容模型使用条数。",
        )
    with c_ncf:
        train_rows_ncf = st.number_input(
            "NCF条数",
            min_value=1000,
            max_value=dynamic_max_rows,
            value=int(st.session_state.get("force_retrain_rows_ncf_hybrid", CF_SAMPLE_SIZE)),
            step=1000,
            key="force_retrain_rows_ncf_hybrid",
            help="融合页重训时 NCF 使用条数。",
        )
    with c_epochs:
        ncf_epochs = st.number_input(
            "NCF Epoch",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("force_retrain_ncf_epochs_hybrid", 8)),
            step=1,
            key="force_retrain_ncf_epochs_hybrid",
            help="融合页强制重训时 NCF 训练轮数（epoch）。",
        )
    with st.expander("融合页 NCF 高级参数", expanded=False):
        hp1, hp2, hp3, hp4 = st.columns(4)
        with hp1:
            ncf_lr_h = st.number_input("lr", min_value=0.00001, max_value=0.01, value=float(st.session_state.get("force_retrain_ncf_lr_hybrid", 0.0006)), step=0.0001, format="%.5f", key="force_retrain_ncf_lr_hybrid")
        with hp2:
            ncf_emb_dim_h = st.number_input("emb_dim", min_value=8, max_value=256, value=int(st.session_state.get("force_retrain_ncf_emb_dim_hybrid", 64)), step=8, key="force_retrain_ncf_emb_dim_hybrid")
        with hp3:
            ncf_neg_k_h = st.number_input("neg_k", min_value=1, max_value=10, value=int(st.session_state.get("force_retrain_ncf_neg_k_hybrid", 3)), step=1, key="force_retrain_ncf_neg_k_hybrid")
        with hp4:
            ncf_batch_size_h = st.number_input("batch_size", min_value=64, max_value=8192, value=int(st.session_state.get("force_retrain_ncf_batch_size_hybrid", 1024)), step=64, key="force_retrain_ncf_batch_size_hybrid")
        hp5, hp6, hp7 = st.columns(3)
        with hp5:
            ncf_pop_ratio_h = st.slider("popular_negative_ratio", 0.0, 1.0, value=float(st.session_state.get("force_retrain_ncf_pop_ratio_hybrid", 0.67)), step=0.01, key="force_retrain_ncf_pop_ratio_hybrid")
        with hp6:
            ncf_min_epochs_h = st.number_input("min_epochs", min_value=1, max_value=30, value=int(st.session_state.get("force_retrain_ncf_min_epochs_hybrid", 5)), step=1, key="force_retrain_ncf_min_epochs_hybrid")
        with hp7:
            ncf_patience_h = st.number_input("patience", min_value=1, max_value=15, value=int(st.session_state.get("force_retrain_ncf_patience_hybrid", 4)), step=1, key="force_retrain_ncf_patience_hybrid")
    with c_force_btn:
        if st.button("强制重新训练模型", key="force_retrain_hybrid"):
            cancel_idle_prefetch()
            with st.spinner("正在强制重训模型并覆盖保存..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    force=True,
                    train_rows_map={
                        "usercf": int(train_rows_usercf),
                        "itemcf": int(train_rows_itemcf),
                        "svd": int(train_rows_svd),
                        "content": int(train_rows_content),
                        "ncf": int(train_rows_ncf),
                    },
                    ncf_epochs=int(ncf_epochs),
                    ncf_train_config={
                        "lr": float(ncf_lr_h),
                        "emb_dim": int(ncf_emb_dim_h),
                        "neg_k": int(ncf_neg_k_h),
                        "popular_negative_ratio": float(ncf_pop_ratio_h),
                        "min_epochs": int(ncf_min_epochs_h),
                        "early_stop_patience": int(ncf_patience_h),
                        "batch_size": int(ncf_batch_size_h),
                    },
                    scopes=["cf", "content", "ncf"],
                )
            if retrained:
                st.success(retrain_msg)
            else:
                st.info(retrain_msg)
    
    # 权重设置（五路独立）
    st.markdown("### 权重设置（可选）")
    use_custom_weights = st.checkbox("使用自定义权重", value=False)

    try:
        max_user_id = get_max_user_id()
    except Exception:
        max_user_id = get_max_user_id_for_hybrid()
    can_use_ncf = user_id is not None and user_id <= max_user_id
    can_use_ncf_with_history = can_use_ncf or (user_id is not None and len(user_history) > 0)
    auto_weights = default_hybrid_weights_5way(user_id, user_history)
    st.caption(
        "后端自动默认权重："
        f"UserCF={auto_weights.get('usercf', 0):.2f}，"
        f"ItemCF={auto_weights.get('itemcf', 0):.2f}，"
        f"SVD={auto_weights.get('svd', 0):.2f}，"
        f"NCF={auto_weights.get('ncf', 0):.2f}，"
        f"内容={auto_weights.get('content', 0):.2f}"
    )

    if use_custom_weights:
        r1, r2, r3 = st.columns(3)
        with r1:
            w_usercf = st.slider("UserCF", 0.0, 1.0, float(auto_weights.get("usercf", 0.2)), 0.05, key="hybrid_w_usercf")
        with r2:
            w_itemcf = st.slider("ItemCF", 0.0, 1.0, float(auto_weights.get("itemcf", 0.15)), 0.05, key="hybrid_w_itemcf")
        with r3:
            w_svd = st.slider("SVD", 0.0, 1.0, float(auto_weights.get("svd", 0.15)), 0.05, key="hybrid_w_svd")
        r4, r5 = st.columns(2)
        with r4:
            if can_use_ncf_with_history:
                w_ncf = st.slider("NCF", 0.0, 1.0, float(auto_weights.get("ncf", 0.4)), 0.05, key="hybrid_w_ncf")
                if not can_use_ncf and len(user_history) > 0:
                    st.caption("将使用个性化 NCF 模式")
            else:
                st.info("NCF 需历史记录或用户 ID 在训练范围内")
                w_ncf = 0.0
        with r5:
            w_content = st.slider("内容推荐", 0.0, 1.0, float(auto_weights.get("content", 0.1)), 0.05, key="hybrid_w_content")

        total = w_usercf + w_itemcf + w_svd + w_ncf + w_content
        if total <= 0:
            st.warning("权重总和不能为 0，已回退为均分")
            w_usercf = w_itemcf = w_svd = w_ncf = w_content = 0.2
            total = 1.0
        weights = {
            "usercf": w_usercf / total,
            "itemcf": w_itemcf / total,
            "svd": w_svd / total,
            "ncf": w_ncf / total,
            "content": w_content / total,
        }
        st.write(
            f"**归一后权重：** UserCF={weights['usercf']:.2f}，ItemCF={weights['itemcf']:.2f}，SVD={weights['svd']:.2f}，"
            f"NCF={weights['ncf']:.2f}，内容={weights['content']:.2f}"
        )
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
    history_signature = tuple(sorted(set(user_history)))
    weight_sig = "auto"
    if use_custom_weights:
        weight_sig = "|".join(f"{weights.get(k, 0):.3f}" for k in ("usercf", "itemcf", "svd", "ncf", "content"))
    cache = st.session_state.setdefault("hybrid_result_cache", {})
    cache_key = f"{user_id}|{topk}|{history_signature}|{weight_sig}"

    if st.button("获取融合推荐", type="primary"):
        try:
            cancel_idle_prefetch()
            with st.spinner("正在融合多种算法生成推荐..."):
                prefetched = None
                # 只有默认参数下才可直接复用后台预热结果
                if (not use_custom_weights) and topk == DEFAULT_RECOMMEND:
                    prefetched = get_prefetch_result("hybrid", username, user_id, user_history, topk=topk)
                result = prefetched if prefetched else hybrid_recommend(user_id, user_history, topk, weights)
                st.session_state['hybrid_recommend_result'] = result
                cache[cache_key] = result
            
            if result:
                st.success("融合推荐生成完成！")
            else:
                st.warning("未找到推荐结果")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # 显示推荐结果
    if cache_key in cache:
        result = cache[cache_key]
        st.session_state['hybrid_recommend_result'] = result
        st.write("**融合推荐结果：**")
        render_recommendation_results(result, prefix='hybrid', show_listen_button=True)
