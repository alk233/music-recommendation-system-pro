# 冷启动推荐页面
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from src.recommend_utils import popularity_cold_start
from app.utils.ui_components import render_recommendation_results
from app.utils.helpers import download_font_if_needed
from app.utils.data_notes import render_page_metric_note
from config import DATA_FILE, STREAMLIT_DATA_NROWS


@st.cache_resource
def get_wordcloud_figs():
    # 生成词云图
    font_path = download_font_if_needed()
    if not font_path:
        return None, None
    
    try:
        import pandas as pd
        df = pd.read_csv(
            DATA_FILE, usecols=['artist_name', 'title'], nrows=STREAMLIT_DATA_NROWS
        )
        artist_counts = df['artist_name'].value_counts().head(20)
        song_counts = df['title'].value_counts().head(20)
        
        artist_freq = artist_counts.to_dict()
        wc_artist = WordCloud(
            font_path=font_path,
            width=500,
            height=250,
            background_color='white'
        ).generate_from_frequencies(artist_freq)
        
        # 创建图像
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.imshow(wc_artist, interpolation='bilinear')
        ax1.axis('off')
        # 添加边框
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        song_freq = song_counts.to_dict()
        wc_song = WordCloud(
            font_path=font_path,
            width=500,
            height=250,
            background_color='white'
        ).generate_from_frequencies(song_freq)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.imshow(wc_song, interpolation='bilinear')
        ax2.axis('off')
        # 添加边框
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        return fig1, fig2
    except Exception as e:
        st.error(f"生成词云图失败: {str(e)}")
        return None, None


@st.cache_data
def get_top_stats():
    # 获取Top统计数据
    import pandas as pd
    try:
        df = pd.read_csv(
            DATA_FILE, usecols=['artist_name', 'title'], nrows=STREAMLIT_DATA_NROWS
        )
        artist_counts = df['artist_name'].value_counts().head(10)
        song_counts = df['title'].value_counts().head(10)
        return artist_counts, song_counts
    except Exception as e:
        st.error(f"统计数据计算失败: {str(e)}")
        return None, None


@st.cache_data
def get_artist_hot_table():
    # 歌手热度表：按歌手聚合，供“歌手热门度榜单”使用
    import pandas as pd

    df = pd.read_csv(
        DATA_FILE,
        usecols=["artist_name", "artist_hotttnesss", "play_count"],
        nrows=STREAMLIT_DATA_NROWS,
    )
    g = (
        df.groupby("artist_name", as_index=False)
        .agg(
            artist_hotttnesss=("artist_hotttnesss", "mean"),
            play_sum=("play_count", "sum"),
        )
        .fillna(0)
    )
    g = g.sort_values(["artist_hotttnesss", "play_sum"], ascending=[False, False])
    return g


def get_artist_hot_rankings(topk=10, seed=None):
    # 从候选Top100歌手中随机抽样topk，保证刷新有变化
    table = get_artist_hot_table()
    if table.empty:
        return []
    candidate_n = min(100, len(table))
    candidates = table.head(candidate_n).copy()
    if len(candidates) <= topk:
        picked = candidates
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=topk, replace=False)
        picked = candidates.iloc[idx]
    picked = picked.sort_values(["artist_hotttnesss", "play_sum"], ascending=[False, False])
    out = []
    for _, r in picked.iterrows():
        out.append(
            f"{r['artist_name']}，歌手热度 {_format_hot(r['artist_hotttnesss'])}，热度合计 {_format_hot(r['play_sum'])}"
        )
    return out


@st.cache_data
def get_artist_song_top10_strict(topk=10):
    # 原先风格：按歌手热度的歌曲榜单（严格TopN，不随机）
    import pandas as pd

    df = pd.read_csv(
        DATA_FILE,
        usecols=["song", "artist_name", "title", "artist_hotttnesss", "play_count"],
        nrows=STREAMLIT_DATA_NROWS,
    )
    # 按歌曲去重后，依据歌手热度+热度合计排序
    g = (
        df.groupby(["song", "artist_name", "title"], as_index=False)
        .agg(artist_hotttnesss=("artist_hotttnesss", "mean"), play_sum=("play_count", "sum"))
        .fillna(0)
        .sort_values(["artist_hotttnesss", "play_sum"], ascending=[False, False])
    )
    g = g.head(topk)
    out = []
    for _, r in g.iterrows():
        out.append(
            f"{r['artist_name']} - {r['title']} (song_id={int(r['song'])})，歌手热度 {_format_hot(r['artist_hotttnesss'])}"
        )
    return out


def get_artist_song_rankings_refreshable(topk=10, seed=None):
    # 可刷新版本：从歌手热曲候选 Top100 中随机抽样 topk
    import pandas as pd

    df = pd.read_csv(
        DATA_FILE,
        usecols=["song", "artist_name", "title", "artist_hotttnesss", "play_count"],
        nrows=STREAMLIT_DATA_NROWS,
    )
    g = (
        df.groupby(["song", "artist_name", "title"], as_index=False)
        .agg(artist_hotttnesss=("artist_hotttnesss", "mean"), play_sum=("play_count", "sum"))
        .fillna(0)
        .sort_values(["artist_hotttnesss", "play_sum"], ascending=[False, False])
    )
    candidates = g.head(min(100, len(g)))
    if len(candidates) <= topk:
        picked = candidates
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=topk, replace=False)
        picked = candidates.iloc[idx]
    picked = picked.sort_values(["artist_hotttnesss", "play_sum"], ascending=[False, False])
    out = []
    for _, r in picked.iterrows():
        out.append(
            f"{r['artist_name']} - {r['title']} (song_id={int(r['song'])})，歌手热度 {_format_hot(r['artist_hotttnesss'])}"
        )
    return out


@st.cache_data
def get_artist_top20_hot():
    # 歌手热度 Top20（仅歌手）
    tab = get_artist_hot_table()
    top = tab.head(20)
    lines = []
    for _, r in top.iterrows():
        lines.append(f"{r['artist_name']}，歌手热度 {_format_hot(r['artist_hotttnesss'])}，热度合计 {_format_hot(r['play_sum'])}")
    return lines


def _format_hot(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "0.0000"


def render():
    # 渲染冷启动推荐页面
    st.title("首页 - 多维度榜单推荐（新用户）")
    render_page_metric_note("home")

    st.markdown("### 榜单设置")
    topk = st.slider(
        "推荐TopN",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

    def _ensure_chart(chart_type: str):
        key = f"cold_start_result_{chart_type}"
        topk_key = f"cold_start_topk_{chart_type}"
        if key not in st.session_state or st.session_state.get(topk_key) != topk:
            import random

            seed = random.randint(0, 1000000)
            st.session_state[key] = popularity_cold_start(
                topk, chart_type=chart_type, diversity=True, seed=seed
            )
            st.session_state[topk_key] = topk
            st.session_state[f"cold_start_seed_{chart_type}"] = seed

    def _refresh_chart(chart_type: str):
        import random

        seed = random.randint(0, 1000000)
        st.session_state[f"cold_start_result_{chart_type}"] = popularity_cold_start(
            topk, chart_type=chart_type, diversity=True, seed=seed
        )
        st.session_state[f"cold_start_topk_{chart_type}"] = topk
        st.session_state[f"cold_start_seed_{chart_type}"] = seed

    _ensure_chart("popularity")

    st.write("---")
    left, right = st.columns(2)

    with left:
        st.markdown("### 🔥 热门度榜单")
        st.caption("按热度合计排序，展示全局最热门歌曲。")
        if st.button("刷新热门度榜单", key="refresh_popularity", use_container_width=True):
            _refresh_chart("popularity")
            st.rerun()
        render_recommendation_results(
            st.session_state.get("cold_start_result_popularity", []),
            prefix='cold_popularity',
            show_listen_button=True
        )

    with right:
        st.markdown("### 🎤 歌手热门度榜单")
        st.caption("从 Top100 候选中抽样，展示歌手热曲榜。")
        st.markdown("#### 歌手热曲榜")
        if ("cold_start_result_artist_song_refresh" not in st.session_state
            or st.session_state.get("cold_start_topk_artist_song_refresh") != topk):
            import random
            seed = random.randint(0, 1000000)
            st.session_state["cold_start_result_artist_song_refresh"] = get_artist_song_rankings_refreshable(topk=topk, seed=seed)
            st.session_state["cold_start_topk_artist_song_refresh"] = topk
            st.session_state["cold_start_seed_artist_song_refresh"] = seed
        if st.button("刷新歌手热曲榜", key="refresh_artist_song", use_container_width=True):
            import random
            seed = random.randint(0, 1000000)
            st.session_state["cold_start_result_artist_song_refresh"] = get_artist_song_rankings_refreshable(topk=topk, seed=seed)
            st.session_state["cold_start_topk_artist_song_refresh"] = topk
            st.session_state["cold_start_seed_artist_song_refresh"] = seed
            st.rerun()
        strict_lines = st.session_state.get("cold_start_result_artist_song_refresh", [])
        render_recommendation_results(
            strict_lines,
            prefix='cold_artist_refreshable',
            show_listen_button=True
        )

    # 全宽展示歌手热度榜单，避免右下角堆叠、左下角空白
    st.write("---")
    st.markdown("### 🎙️ 歌手热度Top20（仅歌手）")
    artist_lines = get_artist_top20_hot()
    left_list = artist_lines[:10]
    right_list = artist_lines[10:20]
    c_left, c_right = st.columns(2)
    with c_left:
        for i, line in enumerate(left_list, 1):
            st.write(f"{i}. {line}")
    with c_right:
        for i, line in enumerate(right_list, 11):
            st.write(f"{i}. {line}")
    
    # 词云和统计可视化
    st.write("---")
    st.markdown("## 最受欢迎的歌手/歌曲词云 Top 20")
    
    fig1, fig2 = get_wordcloud_figs()
    if fig1 and fig2:
        wc_left, wc_right = st.columns(2)
        with wc_left:
            st.markdown(
                """
                <div style="padding:10px 12px;border:1px solid #e8e8e8;border-radius:10px;background:#fafafa;">
                  <h4 style="margin:0 0 6px 0;">🎤 最受欢迎歌手词云</h4>
                  <p style="margin:0 0 8px 0;color:#666;font-size:13px;">词频越高，字体越大。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(fig1, use_container_width=True)
        with wc_right:
            st.markdown(
                """
                <div style="padding:10px 12px;border:1px solid #e8e8e8;border-radius:10px;background:#fafafa;">
                  <h4 style="margin:0 0 6px 0;">🎵 最受欢迎歌曲词云</h4>
                  <p style="margin:0 0 8px 0;color:#666;font-size:13px;">词频越高，字体越大。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(fig2, use_container_width=True)
    else:
        st.warning("词云图生成失败，请检查数据文件")

    # 柱状图可视化
    artist_counts, song_counts = get_top_stats()
    if artist_counts is not None and song_counts is not None:
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top10 最受欢迎歌手")
            st.bar_chart(artist_counts)
        with col2:
            st.subheader("Top10 最受欢迎歌曲")
            st.bar_chart(song_counts)
