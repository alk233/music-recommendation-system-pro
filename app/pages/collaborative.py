# 协同过滤推荐页面
import streamlit as st
from src.recommend_utils import usercf_topn, itemcf_topn, svd_topn
from app.utils.ui_components import render_recommendation_results
from config import MIN_RECOMMEND, MAX_RECOMMEND, DEFAULT_RECOMMEND, CF_MIN_EST_SCORE, CF_SAMPLE_SIZE, DATA_FILE
from app.utils.data_notes import render_page_metric_note
from app.utils.background_prefetch import cancel_idle_prefetch
from app.utils.helpers import (
    maybe_retrain_models_on_user_data_change,
    get_user_data_change_status,
    get_data_total_rows,
    get_last_model_train_params,
    cf_train_params_disk_tick,
)


@st.cache_data(show_spinner=False)
def _estimate_cf_matrix(nrows: int):
    import pandas as pd

    df = pd.read_csv(DATA_FILE, nrows=int(nrows), usecols=["user", "song"])
    n_users = int(df["user"].nunique())
    n_songs = int(df["song"].nunique())
    return {"nrows": int(len(df)), "n_users": n_users, "n_songs": n_songs}


def _fmt_gib(n: float) -> str:
    if n <= 0:
        return "0.00 GiB"
    return f"{n:.2f} GiB"


def render():
    # 渲染协同过滤推荐页面
    st.title("个性化推荐排行榜（协同过滤）")
    render_page_metric_note("cf")
    user_id = st.session_state.get('user_id', None)
    user_history = st.session_state.get('user_history', [])
    username = st.session_state.get('username', '')
    if user_id is None:
        st.warning("⚠️ 请先登录")
        return

    st.write(f"**当前用户：** {username}（用户ID: {user_id}）")
    st.write(f"**历史记录数量：** {len(user_history)} 条")
    dynamic_max_rows = int(get_data_total_rows())
    changed, change_msg, last_time = get_user_data_change_status()
    st.caption(f"数据变更标记：{change_msg}；上次重训：{last_time}")

    if not user_history:
        st.warning("⚠️ 当前用户暂无历史记录，UserCF / ItemCF / SVD 需要历史数据后才能运行。")
        st.info("请先试听几首歌曲（添加历史记录）后再使用协同过滤推荐。")
        # 清除旧结果，避免无历史时仍显示上次推荐
        st.session_state.pop('cf_recommend_result', None)
        st.session_state.pop('cf_request_meta', None)
        return

    # 与磁盘最近一次重训参数对齐。三路参数独立保存，界面仅显示当前选中模型对应输入框。
    # 离开本页再返回时 Streamlit 可能清空控件 key，用非控件 key 的 _cf_train_rows_pref 做回填来源。
    _last_cf = get_last_model_train_params()
    _disk_tick = cf_train_params_disk_tick()
    _pref_key = "_cf_train_rows_pref"
    if _pref_key not in st.session_state or not isinstance(st.session_state.get(_pref_key), dict):
        st.session_state[_pref_key] = {
            "usercf": int(_last_cf["usercf_rows"]),
            "itemcf": int(_last_cf["itemcf_rows"]),
            "svd": int(_last_cf["svd_rows"]),
        }
    _pref = st.session_state[_pref_key]
    for _k, _dk in (("usercf", "usercf_rows"), ("itemcf", "itemcf_rows"), ("svd", "svd_rows")):
        if _k not in _pref or _pref[_k] is None:
            _pref[_k] = int(_last_cf[_dk])
    if st.session_state.get("_collab_cf_disk_tick") != _disk_tick:
        _pref["usercf"] = int(_last_cf["usercf_rows"])
        _pref["itemcf"] = int(_last_cf["itemcf_rows"])
        _pref["svd"] = int(_last_cf["svd_rows"])
        st.session_state[_pref_key] = _pref
        st.session_state["force_retrain_rows_usercf"] = int(_pref["usercf"])
        st.session_state["force_retrain_rows_itemcf"] = int(_pref["itemcf"])
        st.session_state["force_retrain_rows_svd"] = int(_pref["svd"])
        st.session_state["_collab_cf_disk_tick"] = _disk_tick
    else:
        lo, hi = 1000, int(dynamic_max_rows)
        st.session_state.setdefault("force_retrain_rows_usercf", max(lo, min(hi, int(_pref["usercf"]))))
        st.session_state.setdefault("force_retrain_rows_itemcf", max(lo, min(hi, int(_pref["itemcf"]))))
        st.session_state.setdefault("force_retrain_rows_svd", max(lo, min(hi, int(_pref["svd"]))))

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
    
    history_signature = tuple(sorted(set(user_history)))
    current_meta = (model_type, topk, history_signature)
    last_meta = st.session_state.get('cf_request_meta')
    cache = st.session_state.setdefault("cf_result_cache", {})
    cache_key = f"{user_id}|{model_type}|{topk}|{history_signature}"

    # 历史或参数变化后，提醒用户重新触发（TopN变化会影响输出结果）
    if last_meta is not None and last_meta != current_meta:
        st.info("检测到历史记录或 TopN/模型参数变化，请点击“获取个性化推荐”重新训练并更新结果。")

    selected_scope = {"UserCF": "usercf", "ItemCF": "itemcf", "SVD": "svd"}[model_type]
    selected_key = {
        "usercf": "force_retrain_rows_usercf",
        "itemcf": "force_retrain_rows_itemcf",
        "svd": "force_retrain_rows_svd",
    }[selected_scope]
    selected_label = {
        "usercf": "UserCF训练条数",
        "itemcf": "ItemCF训练条数",
        "svd": "SVD训练条数",
    }[selected_scope]
    # 仅在切换模型时回填该模型的参数，避免切换多次后串值。
    if st.session_state.get("_cf_selected_scope") != selected_scope or selected_key not in st.session_state:
        st.session_state[selected_key] = int(_pref[selected_scope])
    st.session_state["_cf_selected_scope"] = selected_scope

    c_train_rows, c_force_btn = st.columns([2, 1])
    with c_train_rows:
        st.caption("仅显示当前选中模型的训练采样条数；切换模型后可分别调整。未重训前仅为临时值。")
        st.number_input(
            selected_label,
            min_value=1000,
            max_value=dynamic_max_rows,
            step=1000,
            key=selected_key,
        )
        _pref[selected_scope] = int(st.session_state[selected_key])
        st.session_state[_pref_key] = _pref
        st.caption(
            f"当前编辑值：{int(st.session_state[selected_key]):,}；"
            f"上次训练值：{int(_last_cf[selected_scope + '_rows']):,}。"
        )

    train_rows_usercf = int(st.session_state.get("force_retrain_rows_usercf", _last_cf["usercf_rows"]))
    train_rows_itemcf = int(st.session_state.get("force_retrain_rows_itemcf", _last_cf["itemcf_rows"]))
    train_rows_svd = int(st.session_state.get("force_retrain_rows_svd", _last_cf["svd_rows"]))
    selected_rows = {
        "usercf": int(train_rows_usercf),
        "itemcf": int(train_rows_itemcf),
        "svd": int(train_rows_svd),
    }[selected_scope]
    with c_force_btn:
        if st.button("强制重新训练模型", key="cf_force_retrain"):
            cancel_idle_prefetch()
            with st.spinner("正在强制重训并覆盖模型..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    force=True,
                    train_rows_map={selected_scope: int(selected_rows)},
                    scopes=[selected_scope],
                )
            st.success(retrain_msg)
            st.rerun()

    if st.button("获取个性化推荐", type="primary"):
        try:
            cancel_idle_prefetch()
            # 每次点击前仅提示当前所选模型的训练预估，避免三条提示同时出现
            try:
                if model_type == "UserCF":
                    est_usercf = _estimate_cf_matrix(int(train_rows_usercf))
                    nu_u, ni_u = int(est_usercf["n_users"]), int(est_usercf["n_songs"])
                    mem_usercf = (nu_u * nu_u * 8) / (1024**3)
                    st.info(
                        f"UserCF 训练预估：读取交互 {int(est_usercf['nrows'])} 行；"
                        f"唯一用户 {nu_u}，唯一歌曲 {ni_u}。"
                    )
                    st.caption(
                        f"UserCF 相似度矩阵≈{nu_u}×{nu_u}（约 {_fmt_gib(mem_usercf)}，float64 估算）。"
                    )
                elif model_type == "ItemCF":
                    est_itemcf = _estimate_cf_matrix(int(train_rows_itemcf))
                    nu_i, ni_i = int(est_itemcf["n_users"]), int(est_itemcf["n_songs"])
                    mem_itemcf = (ni_i * ni_i * 8) / (1024**3)
                    st.info(
                        f"ItemCF 训练预估：读取交互 {int(est_itemcf['nrows'])} 行；"
                        f"唯一用户 {nu_i}，唯一歌曲 {ni_i}。"
                    )
                    st.caption(
                        f"ItemCF 相似度矩阵≈{ni_i}×{ni_i}（约 {_fmt_gib(mem_itemcf)}，float64 估算）。"
                    )
                else:
                    est_svd = _estimate_cf_matrix(int(train_rows_svd))
                    nu_s, ni_s = int(est_svd["n_users"]), int(est_svd["n_songs"])
                    st.info(
                        f"SVD 训练预估：读取交互 {int(est_svd['nrows'])} 行；"
                        f"唯一用户 {nu_s}，唯一歌曲 {ni_s}。"
                    )
                    st.caption("SVD 为矩阵分解，不构建 UserCF/ItemCF 那种全量相似度矩阵。")
            except Exception as _:
                st.caption("本次 CF 训练预估失败（将直接执行训练）。")

            with st.spinner("检查用户数据变化并自动重训模型..."):
                retrained, retrain_msg = maybe_retrain_models_on_user_data_change(
                    train_rows_map={selected_scope: int(selected_rows)},
                    scopes=[selected_scope],
                )
            if retrained:
                st.success(retrain_msg)
            with st.spinner(f"正在使用{model_type}进行个性化训练并生成推荐..."):
                result = None

                if not result:
                    if model_type == "UserCF":
                        result = usercf_topn(
                            user_id, topk, user_history=user_history, train_rows_override=int(train_rows_usercf)
                        )
                    elif model_type == "ItemCF":
                        result = itemcf_topn(
                            user_id, topk, user_history=user_history, train_rows_override=int(train_rows_itemcf)
                        )
                    else:
                        result = svd_topn(
                            user_id, topk, user_history=user_history, train_rows_override=int(train_rows_svd)
                        )

                st.session_state['cf_recommend_result'] = result
                st.session_state['cf_request_meta'] = current_meta
                cache[cache_key] = result

            if result:
                st.success("个性化推荐生成完成！")
            else:
                st.warning("未找到推荐结果，请增加历史记录后重试。")
        except Exception as e:
            st.error(f"推荐生成失败: {str(e)}")
            st.info("请检查历史记录与数据文件格式是否正确。")

    if cache_key in cache:
        result = cache[cache_key]
        st.session_state['cf_recommend_result'] = result
        st.session_state['cf_request_meta'] = current_meta
        st.write(f"**{model_type} 个性化推荐Top{topk}：**")
        st.caption(f"已启用低分过滤：当前所选模型的原始估计分 < {CF_MIN_EST_SCORE} 的候选不展示。")
        st.caption(
            "名次仅由当前模型的原始估计决定；「推荐值」为在本页已展示候选上平滑映射到约 0～100 的读数，不改变排序。"
        )
        render_recommendation_results(result, prefix='cf', show_listen_button=True)
