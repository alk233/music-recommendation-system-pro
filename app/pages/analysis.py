# 推荐分析页面
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from app.utils.helpers import load_song_info
from config import DATA_FILE


def render():
    # 渲染推荐分析页面
    st.title("📊 必吃榜 - 推荐分析")
    
    # 获取当前用户信息
    username = st.session_state.get('username', '')
    user_history = st.session_state.get('user_history', [])
    
    if not username:
        st.warning("⚠️ 请先登录")
        return
    
    # 分析页面说明
    
    st.write("---")
    
    # 1. 用户画像分析
    st.markdown("## 🎯 用户画像分析")
    
    if user_history:
        # 加载歌曲信息
        df_songs = load_song_info()
        if not df_songs.empty:
            # 获取用户历史歌曲的详细信息
            user_songs = []
            for song_id in user_history:
                if song_id in df_songs.index:
                    user_songs.append(df_songs.loc[song_id])
            
            if user_songs:
                user_df = pd.DataFrame(user_songs)
                
                # 最喜欢的歌手
                st.markdown("### 你最喜欢的歌手（Top 5）")
                if 'artist_name' in user_df.columns:
                    artist_counts = user_df['artist_name'].value_counts().head(5)
                    if len(artist_counts) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(artist_counts)
                        with col2:
                            st.write("**歌手列表：**")
                            for idx, (artist, count) in enumerate(artist_counts.items(), 1):
                                st.write(f"{idx}. {artist} ({count}首)")
                
                # 年份分布
                st.markdown("### 你喜欢的音乐年份分布")
                try:
                    df_full = pd.read_csv(DATA_FILE, usecols=['song', 'year'])
                    user_years = []
                    for song_id in user_history:
                        song_data = df_full[df_full['song'] == song_id]
                        if not song_data.empty and pd.notna(song_data['year'].iloc[0]):
                            user_years.append(int(song_data['year'].iloc[0]))
                    
                    if user_years:
                        year_counts = pd.Series(user_years).value_counts().sort_index()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.line_chart(year_counts)
                        with col2:
                            st.write("**年份统计：**")
                            for year, count in year_counts.items():
                                st.write(f"{year}年: {count}首")
                except:
                    st.info("年份数据不可用")
                
                # 用户画像雷达图
                st.markdown("### 你的音乐偏好画像")
                try:
                    # 设置中文字体
                    import platform
                    system = platform.system()
                    if system == 'Windows':
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
                    elif system == 'Darwin':
                        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
                    else:
                        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
                    
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    df_full = pd.read_csv(DATA_FILE, usecols=['song', 'artist_familiarity', 'artist_hotttnesss'])
                    user_features = []
                    for song_id in user_history:
                        song_data = df_full[df_full['song'] == song_id]
                        if not song_data.empty:
                            user_features.append({
                                'familiarity': song_data['artist_familiarity'].iloc[0] if pd.notna(song_data['artist_familiarity'].iloc[0]) else 0,
                                'hotttnesss': song_data['artist_hotttnesss'].iloc[0] if pd.notna(song_data['artist_hotttnesss'].iloc[0]) else 0
                            })
                    
                    if user_features:
                        features_df = pd.DataFrame(user_features)
                        avg_familiarity = features_df['familiarity'].mean()
                        avg_hotttnesss = features_df['hotttnesss'].mean()
                        
                        # 创建雷达图数据
                        categories = ['歌手熟悉度', '歌手热度']
                        values = [avg_familiarity * 100, avg_hotttnesss * 100]
                        
                        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
                        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                        values += values[:1]
                        angles += angles[:1]
                        
                        ax.plot(angles, values, 'o-', linewidth=2, label='你的偏好')
                        ax.fill(angles, values, alpha=0.25)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories)
                        ax.set_title('音乐偏好雷达图', size=16, fontweight='bold', pad=20)
                        ax.legend()
                        
                        ax.set_ylim(0, 100)
                        ax.grid(True)
                        
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.info(f"画像分析数据不可用: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("你还没有历史记录，无法生成用户画像。请先听一些歌曲！")
    
    st.write("---")
    
    # 2. 热门内容排行
    st.markdown("## 🔥 热门内容排行")
    
    try:
        df = pd.read_csv(DATA_FILE, usecols=['song', 'artist_name', 'title', 'play_count'])
        
        # 最受欢迎的歌手
        st.markdown("### 最受欢迎的歌手（Top 10）")
        artist_stats = df.groupby('artist_name')['play_count'].sum().sort_values(ascending=False).head(10)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(artist_stats)
        with col2:
            st.write("**排行榜：**")
            for idx, (artist, count) in enumerate(artist_stats.items(), 1):
                st.write(f"{idx}. {artist} (播放量: {int(count):,})")
        
        # 最受欢迎的歌曲
        st.markdown("### 最受欢迎的歌曲（Top 10）")
        song_stats = df.groupby(['song', 'artist_name', 'title'])['play_count'].sum().reset_index()
        song_stats = song_stats.sort_values('play_count', ascending=False).head(10)
        song_stats['song_name'] = song_stats['artist_name'] + ' - ' + song_stats['title']
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(song_stats.set_index('song_name')['play_count'])
        with col2:
            st.write("**排行榜：**")
            for idx, row in song_stats.iterrows():
                st.write(f"{song_stats.index.get_loc(idx)+1}. {row['song_name']} (播放量: {int(row['play_count']):,})")
        
    except Exception as e:
        st.error(f"加载热门内容失败: {str(e)}")
    
