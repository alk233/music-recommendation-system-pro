# 冷启动推荐页面
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.recommend_utils import content_cold_start
from app.utils.ui_components import render_recommendation_results
from app.utils.helpers import download_font_if_needed
from config import DATA_FILE


@st.cache_resource
def get_wordcloud_figs():
    # 生成词云图
    font_path = download_font_if_needed()
    if not font_path:
        return None, None
    
    try:
        import pandas as pd
        df = pd.read_csv(DATA_FILE, usecols=['artist_name', 'title'])
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
        df = pd.read_csv(DATA_FILE, usecols=['artist_name', 'title'])
        artist_counts = df['artist_name'].value_counts().head(10)
        song_counts = df['title'].value_counts().head(10)
        return artist_counts, song_counts
    except Exception as e:
        st.error(f"统计数据计算失败: {str(e)}")
        return None, None


def render():
    # 渲染冷启动推荐页面
    st.title("首页 - 内容推荐冷启动")
    
    # 推荐数量选择
    topk = st.slider(
        "推荐TopN",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    
    # 初始化推荐结果
    if 'cold_start_result' not in st.session_state or st.session_state.get('cold_start_topk', 10) != topk:
        st.session_state['cold_start_result'] = content_cold_start(topk)
        st.session_state['cold_start_topk'] = topk
    
    # 刷新按钮
    if st.button("刷新冷启动推荐"):
        st.session_state['cold_start_result'] = content_cold_start(topk)
        st.session_state['cold_start_topk'] = topk
        st.success("推荐已刷新！")
        st.rerun()
    
    # 显示推荐结果
    result = st.session_state['cold_start_result']
    st.write("**为新用户推荐的歌曲：**")
    
    # 显示推荐结果
    render_recommendation_results(result, prefix='cold', show_listen_button=True)
    
    # 词云和统计可视化
    st.write("---")
    st.markdown("## 最受欢迎的歌手/歌曲词云 Top 20")
    
    fig1, fig2 = get_wordcloud_figs()
    if fig1 and fig2:
        # 单独居中显示标题
        st.markdown("<h3 style='text-align: center;'>最受欢迎的歌手词云</h3>", unsafe_allow_html=True)
        # 不使用容器宽度，保持图像较小
        st.pyplot(fig1, use_container_width=False)
        
        # 单独居中显示标题
        st.markdown("<h3 style='text-align: center;'>最受欢迎的歌曲词云</h3>", unsafe_allow_html=True)
        st.pyplot(fig2, use_container_width=False)
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
