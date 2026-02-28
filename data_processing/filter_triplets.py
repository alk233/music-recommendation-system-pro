"""
过滤活跃用户和热门歌曲
"""
import sys
import os
import pandas as pd

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import TRIPLETS_FILE, FILTERED_TRIPLETS, USER_MIN_PLAYS, SONG_MIN_PLAYS


def filter_triplets():
    """过滤活跃用户和热门歌曲"""
    print('正在读取train_triplets.txt...')
    df = pd.read_csv(
        TRIPLETS_FILE,
        sep='\t',
        names=['user_id', 'song_id', 'play_count'],
        dtype={'user_id': 'object', 'song_id': 'object', 'play_count': 'int32'}
    )
    print(f'原始数据: {len(df)} 条记录')
    
    # 统计每个用户的总播放量
    print('正在统计用户播放量...')
    user_play_counts = df.groupby('user_id')['play_count'].sum()
    active_users = user_play_counts[user_play_counts >= USER_MIN_PLAYS].index
    print(f'活跃用户（播放量>={USER_MIN_PLAYS}）: {len(active_users)} 个')
    
    # 统计每首歌的总播放量
    print('正在统计歌曲播放量...')
    song_play_counts = df.groupby('song_id')['play_count'].sum()
    popular_songs = song_play_counts[song_play_counts >= SONG_MIN_PLAYS].index
    print(f'热门歌曲（播放量>={SONG_MIN_PLAYS}）: {len(popular_songs)} 首')
    
    # 过滤数据
    print('正在过滤数据...')
    filtered = df[df['user_id'].isin(active_users) & df['song_id'].isin(popular_songs)]
    print(f'过滤后数据: {len(filtered)} 条记录')
    
    # 保存结果
    filtered.to_csv(FILTERED_TRIPLETS, index=False)
    print(f'过滤后的数据已保存为 {FILTERED_TRIPLETS}')


if __name__ == '__main__':
    filter_triplets()
