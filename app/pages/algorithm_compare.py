# 算法对比与离线评测专页
import os
import random
import pickle
import json

import streamlit as st
import pandas as pd
import altair as alt

from app.utils.data_notes import render_page_metric_note, render_offline_metrics_glossary
from app.utils.helpers import get_last_model_train_params
from config import (
    CF_SAMPLE_SIZE,
    CONTENT_SAMPLE_SIZE,
    NCF_SAMPLE_SIZE,
    DEFAULT_RECOMMEND,
    NCF_MODEL_PATH,
    CF_META_PATH,
    CONTENT_MODEL_BUNDLE_PATH,
)
from evaluation.compare_algorithms import run_algorithm_comparison

_CORE_METRICS = ["HR(%)", "NDCG(%)", "MRR(%)", "F1(%)", "准确率(%)"]
_CORE_OPTIONAL = ["候选AUC"]
_METRIC_COLOR = {
    "HR(%)": "#1DB954",   # 绿色：命中
    "NDCG(%)": "#4EA1FF", # 蓝色：排序质量
    "MRR(%)": "#A56EFF",  # 紫色：首位命中能力
    "F1(%)": "#FFB347",
    "准确率(%)": "#F66D9B",
}


def _prepare_metric_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "准确率(%)" not in out.columns:
        if "P(%)" in out.columns:
            out["准确率(%)"] = out["P(%)"]
        elif "P@10" in out.columns:
            out["准确率(%)"] = out["P@10"] * 100.0
    return out


def _format_metric_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """表格展示用：核心指标固定两位小数，避免看起来像被取整。"""
    out = df.copy()
    two_dec_cols = [c for c in _CORE_METRICS if c in out.columns]
    for col in two_dec_cols:
        out[col] = out[col].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    if "候选AUC" in out.columns:
        out["候选AUC"] = out["候选AUC"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "—")
    return out


def _pros_cons_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "算法": "UserCF",
                "思路": "找相似用户，借用其偏好",
                "优势": "同类兴趣迁移直观，实现成熟",
                "局限": "用户稀疏或新用户时效果下降，易受热门牵引",
            },
            {
                "算法": "ItemCF",
                "思路": "物品共现与相似度",
                "优势": "物品关系相对稳定，适合用户兴趣较宽",
                "局限": "新物品缺少历史时偏弱，计算量随物品增长",
            },
            {
                "算法": "SVD",
                "思路": "低维隐因子分解评分矩阵",
                "优势": "泛化能力较好，可缓解稀疏",
                "局限": "可解释性弱，需训练与调参",
            },
            {
                "算法": "内容推荐",
                "思路": "歌曲元数据与用户画像相似度",
                "优势": "不依赖大量共现，利于冷启动与可解释",
                "局限": "难捕捉从众与隐性协同信号",
            },
            {
                "算法": "NCF",
                "思路": "神经网络拟合用户–物品交互",
                "优势": "可表达非线性关系",
                "局限": "依赖样本量与算力，可解释性弱",
            },
            {
                "算法": "融合推荐",
                "思路": "UserCF、ItemCF、SVD、NCF 与内容五路分别校准后加权",
                "优势": "综合多类归纳偏置，线上与离线评测公式对齐",
                "局限": "五路权重需调；算力开销高于单路；候选AUC 与 HR 量纲不同勿混读",
            },
            {
                "算法": "热门基线",
                "思路": "按全局热度排序",
                "优势": "作对照下界与热门策略参照",
                "局限": "无个性化",
            },
            {
                "算法": "随机基线",
                "思路": "候选随机排列",
                "优势": "检验指标是否显著优于随机",
                "局限": "无实用价值",
            },
        ]
    )


def _pick_algo_row(df: pd.DataFrame, algo_name: str):
    if df is None or df.empty:
        return None
    hit = df[df["算法"] == algo_name]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def render():
    # 首次进入页面时，优先恢复上次运行时参数
    last_params = st.session_state.get("algorithm_compare_last_params", {})
    default_k = int(last_params.get("k", DEFAULT_RECOMMEND))
    default_mu = int(last_params.get("max_users", 30))
    default_ms = int(last_params.get("max_samples", 150))
    default_nr = int(last_params.get("nrows", min(CF_SAMPLE_SIZE, 100000)))
    default_ncf = bool(last_params.get("include_ncf", os.path.isfile(NCF_MODEL_PATH)))
    default_hybrid = bool(last_params.get("include_hybrid", True))
    default_seed = int(last_params.get("fixed_seed", 42))
    default_hybrid_custom = bool(last_params.get("use_custom_hybrid", False))
    leg_cf = float(last_params.get("w_cf", 0.5))
    default_w_usercf = float(last_params.get("w_usercf", 0.20))
    default_w_itemcf = float(last_params.get("w_itemcf", 0.15))
    default_w_svd = float(last_params.get("w_svd", 0.15))
    default_w_ncf = float(last_params.get("w_ncf", 0.40))
    default_w_ct = float(last_params.get("w_content", 0.10))
    default_nonuniform = bool(last_params.get("enable_nonuniform_eval", True))
    default_fair = bool(last_params.get("enable_fair_eval", False))
    default_fair_nrows = int(last_params.get("fair_nrows", 100000))
    last_model_params = get_last_model_train_params()
    default_nonuniform_usercf = int(last_model_params.get("usercf_rows", last_model_params.get("cf_rows", min(CF_SAMPLE_SIZE, 100000))))
    default_nonuniform_itemcf = int(last_model_params.get("itemcf_rows", last_model_params.get("cf_rows", min(CF_SAMPLE_SIZE, 100000))))
    default_nonuniform_svd = int(last_model_params.get("svd_rows", last_model_params.get("cf_rows", min(CF_SAMPLE_SIZE, 100000))))
    default_nonuniform_content = int(last_model_params.get("content_rows", min(CONTENT_SAMPLE_SIZE, 100000)))
    default_nonuniform_ncf = int(last_model_params.get("ncf_rows", min(NCF_SAMPLE_SIZE, 100000)))
    default_ncf_epochs = int(last_model_params.get("ncf_epochs", 8))

    st.session_state["cmp_k"] = default_k
    st.session_state.setdefault("cmp_mu", default_mu)
    st.session_state.setdefault("cmp_ms", default_ms)
    st.session_state.setdefault("cmp_nr", default_nr)
    st.session_state.setdefault("cmp_ncf", default_ncf)
    st.session_state.setdefault("cmp_hybrid", default_hybrid)
    st.session_state.setdefault("cmp_seed", default_seed)
    st.session_state.setdefault("cmp_hybrid_custom", default_hybrid_custom)
    st.session_state.setdefault("cmp_w_usercf", default_w_usercf)
    st.session_state.setdefault("cmp_w_itemcf", default_w_itemcf)
    st.session_state.setdefault("cmp_w_svd", default_w_svd)
    st.session_state.setdefault("cmp_w_ncf", default_w_ncf)
    st.session_state.setdefault("cmp_w_ct", default_w_ct)
    st.session_state.setdefault("cmp_nonuniform", default_nonuniform)
    st.session_state.setdefault("cmp_fair_eval", default_fair)
    st.session_state.setdefault("cmp_fair_nrows", default_fair_nrows)

    st.title("算法对比与评测")
    st.caption("统一协议下的多指标离线对比，以及各算法适用场景对照")
    render_page_metric_note("compare")
    render_offline_metrics_glossary(in_expander=False)

    st.markdown("### 算法特点对照")
    st.dataframe(_pros_cons_table(), use_container_width=True, hide_index=True)

    st.write("---")
    st.markdown("### 离线跑分")
    st.caption(
        f"说明：该页面以“工程效果”为目标，各模型允许使用不同训练数据量；指标仍按统一评测协议计算。"
        f"配置默认值：CF_SAMPLE_SIZE={CF_SAMPLE_SIZE}，CONTENT_SAMPLE_SIZE={CONTENT_SAMPLE_SIZE}，"
        f"NCF_SAMPLE_SIZE={NCF_SAMPLE_SIZE}。实际持久化模型训练规模以“最近一次重训参数/模型元数据”为准。"
    )
    st.caption(
        f"最近一次重训参数：UserCF={default_nonuniform_usercf}，ItemCF={default_nonuniform_itemcf}，"
        f"SVD={default_nonuniform_svd}，内容={default_nonuniform_content}，NCF={default_nonuniform_ncf}（NCF Epoch={default_ncf_epochs}）。"
    )
    if os.path.isfile(CF_META_PATH):
        try:
            with open(CF_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.caption(
                f"CF 持久化模型数据量：rows={meta.get('n_rows', 'N/A')}，"
                f"users={meta.get('n_users', 'N/A')}，songs={meta.get('n_songs', 'N/A')}。"
            )
        except Exception:
            pass
    if os.path.isfile(CONTENT_MODEL_BUNDLE_PATH):
        try:
            with open(CONTENT_MODEL_BUNDLE_PATH, "rb") as f:
                content_bundle = pickle.load(f)
            cmeta = content_bundle.get("meta", {})
            st.caption(f"内容模型训练时 n_songs（歌曲覆盖数）={cmeta.get('n_songs', 'N/A')}。")
        except Exception:
            pass
    st.markdown("可同时运行两套结果：工程效果对比（读已有模型）与同规模公平对比（固定行数现场训练）。")

    with st.expander("评测参数", expanded=False):
        eval_k = int(DEFAULT_RECOMMEND)
        st.caption(f"TopK 固定：{eval_k}（按默认推荐数量）")
        eval_max_users = st.slider("参与划分的最大用户数（建议30）", 10, 250, 30, 10, key="cmp_mu")
        eval_max_samples = st.slider("最大测试样本数", 50, 500, 150, 10, key="cmp_ms")
        include_ncf = True
        st.caption("NCF：默认纳入")
        include_hybrid = st.checkbox("纳入融合推荐", value=True, key="cmp_hybrid")
        use_custom_hybrid = st.checkbox("融合使用自定义权重", value=False, key="cmp_hybrid_custom")
        if use_custom_hybrid:
            a1, a2, a3 = st.columns(3)
            with a1:
                w_usercf = st.slider("融合-UserCF", 0.0, 1.0, float(st.session_state.get("cmp_w_usercf", default_w_usercf)), 0.05, key="cmp_w_usercf")
            with a2:
                w_itemcf = st.slider("融合-ItemCF", 0.0, 1.0, float(st.session_state.get("cmp_w_itemcf", default_w_itemcf)), 0.05, key="cmp_w_itemcf")
            with a3:
                w_svd = st.slider("融合-SVD", 0.0, 1.0, float(st.session_state.get("cmp_w_svd", default_w_svd)), 0.05, key="cmp_w_svd")
            b1, b2 = st.columns(2)
            with b1:
                w_ncf = st.slider("融合-NCF", 0.0, 1.0, float(st.session_state.get("cmp_w_ncf", default_w_ncf)), 0.05, key="cmp_w_ncf")
            with b2:
                w_content = st.slider("融合-内容", 0.0, 1.0, float(st.session_state.get("cmp_w_ct", default_w_ct)), 0.05, key="cmp_w_ct")
            w_sum = w_usercf + w_itemcf + w_svd + w_ncf + w_content
            if w_sum <= 0:
                w_usercf, w_itemcf, w_svd, w_ncf, w_content = 0.20, 0.15, 0.15, 0.40, 0.10
                w_sum = 1.0
            hybrid_weights = {
                "usercf": w_usercf / w_sum,
                "itemcf": w_itemcf / w_sum,
                "svd": w_svd / w_sum,
                "ncf": w_ncf / w_sum,
                "content": w_content / w_sum,
            }
        else:
            hybrid_weights = None
        fixed_seed = st.number_input("固定随机种子", min_value=1, max_value=999999, value=42, step=1, key="cmp_seed")

        enable_nonuniform_eval = st.checkbox("运行工程效果对比（读取已有模型）", value=True, key="cmp_nonuniform")
        enable_fair_eval = st.checkbox("运行同规模公平对比（固定10w并现场训练）", value=False, key="cmp_fair_eval")
        fair_nrows = int(st.number_input("同规模训练行数（默认10w）", min_value=1000, step=1000, key="cmp_fair_nrows"))
        st.caption(
            "工程对比模型来源（最近一次重训参数）："
            f"UserCF={default_nonuniform_usercf}，ItemCF={default_nonuniform_itemcf}，"
            f"SVD={default_nonuniform_svd}，内容={default_nonuniform_content}，NCF={default_nonuniform_ncf}（NCF Epoch={default_ncf_epochs}）。"
        )
        st.caption("工程对比不在此页重训模型；同规模对比会在本次运行中按固定行数现场训练。")

    if st.button("运行离线对比", type="primary", key="cmp_run"):
        try:
            run_seed = int(fixed_seed)
            result_engineering = None
            result_fair = None
            with st.spinner("执行离线评测，请稍候…"):
                if enable_nonuniform_eval:
                    result_engineering = run_algorithm_comparison(
                        nrows=int(fair_nrows),
                        min_interactions=8, test_ratio=0.2, max_users=int(eval_max_users), max_samples=int(eval_max_samples),
                        negatives=99, topk=int(eval_k), with_ncf=include_ncf, include_hybrid=include_hybrid,
                        hybrid_weights=hybrid_weights, seed=run_seed, include_baselines=True,
                        use_persisted_models=True,
                    )
                    result_engineering["meta"]["protocol"] = "工程效果对比：读取当前已持久化模型（不在此页重训）。"
                    result_engineering["meta"]["model_source"] = {
                        "usercf_rows": int(default_nonuniform_usercf),
                        "itemcf_rows": int(default_nonuniform_itemcf),
                        "svd_rows": int(default_nonuniform_svd),
                        "content_rows": int(default_nonuniform_content),
                        "ncf_rows": int(default_nonuniform_ncf),
                    }
                if enable_fair_eval:
                    result_fair = run_algorithm_comparison(
                        nrows=int(fair_nrows),
                        min_interactions=8, test_ratio=0.2, max_users=int(eval_max_users), max_samples=int(eval_max_samples),
                        negatives=99, topk=int(eval_k), with_ncf=include_ncf, include_hybrid=include_hybrid,
                        hybrid_weights=hybrid_weights, seed=run_seed, include_baselines=True,
                        use_persisted_models=False,
                    )

            st.session_state["algorithm_compare_eval"] = result_engineering
            st.session_state["algorithm_compare_eval_fair"] = result_fair
            st.session_state["algorithm_compare_seed"] = run_seed
            st.session_state["algorithm_compare_last_params"] = {
                "k": int(eval_k),
                "max_users": int(eval_max_users),
                "max_samples": int(eval_max_samples),
                "include_ncf": bool(include_ncf),
                "include_hybrid": bool(include_hybrid),
                "use_custom_hybrid": bool(use_custom_hybrid),
                "w_usercf": float(st.session_state.get("cmp_w_usercf", default_w_usercf)),
                "w_itemcf": float(st.session_state.get("cmp_w_itemcf", default_w_itemcf)),
                "w_svd": float(st.session_state.get("cmp_w_svd", default_w_svd)),
                "w_ncf": float(st.session_state.get("cmp_w_ncf", default_w_ncf)),
                "w_content": float(st.session_state.get("cmp_w_ct", default_w_ct)),
                "fixed_seed": int(fixed_seed),
                "enable_nonuniform_eval": bool(enable_nonuniform_eval),
                "enable_fair_eval": bool(enable_fair_eval),
                "fair_nrows": int(fair_nrows),
                "nrows": int(fair_nrows),
            }
            st.success("评测完成")
        except Exception as e:
            st.error(f"评测失败：{e}")
            import traceback

            st.code(traceback.format_exc())

    ev = st.session_state.get("algorithm_compare_eval")
    if ev:
        meta = ev["meta"]
        if st.session_state.get("algorithm_compare_last_params"):
            st.caption("已自动恢复上一次测评参数与结果。")
        st.info(meta.get("protocol", "工程效果对比结果（分模型训练行数）。"))
        st.caption(f"本次随机种子: {st.session_state.get('algorithm_compare_seed', 'N/A')}")
        ms = meta.get("model_source", {}) if isinstance(meta.get("model_source"), dict) else {}
        if ms:
            st.caption(
                "工程对比模型来源（最近持久化模型参数）："
                f"UserCF={ms.get('usercf_rows', 'N/A')}，ItemCF={ms.get('itemcf_rows', 'N/A')}，"
                f"SVD={ms.get('svd_rows', 'N/A')}，内容={ms.get('content_rows', 'N/A')}，NCF={ms.get('ncf_rows', 'N/A')}。"
            )
        st.caption(
            f"评测样本口径：候选集={meta.get('candidate_size', 'N/A')}（1 正例 + {meta.get('negatives', 'N/A')} 负例），"
            f"有效样本={meta.get('effective_samples', 'N/A')}，跳过={meta.get('skipped', 'N/A')}。"
        )
        st.caption("工程对比基线口径：基于本次评测切分得到的训练集热度与随机排序。")
        if meta.get("hybrid_weights") is not None:
            hw = meta["hybrid_weights"]
            st.caption(
                f"评测用融合权重：UserCF={hw.get('usercf',0):.2f}，ItemCF={hw.get('itemcf',0):.2f}，SVD={hw.get('svd',0):.2f}，"
                f"NCF={hw.get('ncf',0):.2f}，内容={hw.get('content',0):.2f}。"
                "融合分：候选内五路原始分分别鲁棒校准后按上式加权，与线上一键融合公式一致；融合行另报告候选AUC。"
            )
        else:
            st.caption("融合未纳入本次评测。")
        if meta.get("ncf_error"):
            st.warning(f"NCF 未加载：{meta['ncf_error']}")

        st.markdown("#### 主算法")
        st.caption("面向展示：HR、NDCG、MRR、F1、准确率；融合行另含候选AUC（ROC AUC，见离线指标释义表）。")
        df_main = _prepare_metric_view(ev["df_main"].copy())
        df_main_show = _format_metric_table_for_display(df_main)
        show_cols = ["算法"] + [c for c in _CORE_METRICS if c in df_main.columns]
        show_cols += [c for c in _CORE_OPTIONAL if c in df_main.columns]
        st.dataframe(df_main_show[show_cols], use_container_width=True, hide_index=True)

        chart_cols = [c for c in _CORE_METRICS if c in df_main.columns]
        if chart_cols:
            st.markdown("#### 指标对比图")
            plot_df = df_main[["算法"] + chart_cols].dropna(how="all")
            if not plot_df.empty:
                long_df = plot_df.melt(
                    id_vars=["算法"],
                    value_vars=chart_cols,
                    var_name="指标",
                    value_name="得分",
                )
                order = (
                    plot_df.sort_values("NDCG(%)", ascending=False)["算法"].tolist()
                    if "NDCG(%)" in plot_df.columns
                    else plot_df["算法"].tolist()
                )
                chart = (
                    alt.Chart(long_df)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        y=alt.Y("算法:N", sort=order, title=None),
                        x=alt.X("得分:Q", title="分数（%）", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color(
                            "指标:N",
                            scale=alt.Scale(
                                domain=list(_METRIC_COLOR.keys()),
                                range=list(_METRIC_COLOR.values()),
                            ),
                            legend=alt.Legend(title=None, orient="top"),
                        ),
                        yOffset=alt.YOffset("指标:N"),
                        tooltip=[
                            alt.Tooltip("算法:N"),
                            alt.Tooltip("指标:N"),
                            alt.Tooltip("得分:Q", format=".2f"),
                        ],
                    )
                    .properties(height=360)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(
                        grid=True,
                        gridOpacity=0.15,
                        tickColor="#666666",
                        domainColor="#666666",
                        labelColor="#D8D8D8",
                        titleColor="#D8D8D8",
                    )
                    .configure_legend(labelColor="#D8D8D8")
                    .configure(background="#101418")
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption("按 NDCG 从高到低排序，颜色分别对应 HR、NDCG、MRR。")

        df_base = ev.get("df_baseline")
        if df_base is not None and not df_base.empty:
            st.markdown("#### 基线")
            df_base = _prepare_metric_view(df_base.copy())
            df_base_show = _format_metric_table_for_display(df_base)
            show_cols = ["算法"] + [c for c in _CORE_METRICS if c in df_base.columns]
            st.dataframe(df_base_show[show_cols], use_container_width=True, hide_index=True)

    ev_fair = st.session_state.get("algorithm_compare_eval_fair")
    if ev_fair:
        st.write("---")
        st.markdown("### 同规模公平对比")
        meta = ev_fair.get("meta", {})
        st.info(meta.get("protocol", "同规模对比结果（统一训练行数）。"))
        st.caption(
            f"同规模口径：训练行数={meta.get('input_nrows', 'N/A')}，"
            f"训练用户={meta.get('train_unique_users', 'N/A')}，训练歌曲={meta.get('train_unique_songs', 'N/A')}。"
        )
        st.caption(
            f"候选集={meta.get('candidate_size', 'N/A')}（1 正例 + {meta.get('negatives', 'N/A')} 负例），"
            f"有效样本={meta.get('effective_samples', 'N/A')}，跳过={meta.get('skipped', 'N/A')}。"
        )
        df_main = _prepare_metric_view(ev_fair["df_main"].copy())
        df_main_show = _format_metric_table_for_display(df_main)
        show_cols = ["算法"] + [c for c in _CORE_METRICS if c in df_main.columns]
        show_cols += [c for c in _CORE_OPTIONAL if c in df_main.columns]
        st.dataframe(df_main_show[show_cols], use_container_width=True, hide_index=True)
        df_base = ev_fair.get("df_baseline")
        if df_base is not None and not df_base.empty:
            st.markdown("#### 基线（同规模）")
            df_base = _prepare_metric_view(df_base.copy())
            df_base_show = _format_metric_table_for_display(df_base)
            show_cols = ["算法"] + [c for c in _CORE_METRICS if c in df_base.columns]
            st.dataframe(df_base_show[show_cols], use_container_width=True, hide_index=True)
