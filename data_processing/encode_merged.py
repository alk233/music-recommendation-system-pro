"""
对合并后的数据进行编码
"""
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import FILTERED_TRIPLETS, SONGS_CSV, FINAL_MERGED_ENCODED


def encode_merged():
    """对合并后的数据进行编码"""
    # 1. 读取过滤后的行为数据和歌曲元数据
    print('正在读取filtered_triplets.csv...')
    df = pd.read_csv(FILTERED_TRIPLETS)
    print(f'读取到 {len(df)} 条行为数据')
    
    print('正在读取songs.csv...')
    songs = pd.read_csv(SONGS_CSV)
    print(f'读取到 {len(songs)} 条歌曲元数据')
    
    # 2. 合并（用song_id作为键）
    print('正在合并数据...')
    merged = pd.merge(df, songs, left_on='song_id', right_on='song_id', how='left')
    print(f'合并后数据: {len(merged)} 条记录')
    
    # 3. 对user_id和song_id编码
    print('正在对user_id和song_id编码...')
    user_encoder = LabelEncoder()
    song_encoder = LabelEncoder()
    merged['user'] = user_encoder.fit_transform(merged['user_id'])
    merged['song'] = song_encoder.fit_transform(merged['song_id'])
    print(f'用户数量: {merged["user"].nunique()}, 歌曲数量: {merged["song"].nunique()}')
    
    # 4. 选择并重命名字段
    print('正在选择字段...')
    final = merged[[
        'user', 'song', 'play_count', 'title', 'release',
        'artist_name', 'artist_familiarity', 'artist_hotttnesss', 'year'
    ]]
    
    # 5. 保存结果
    final.to_csv(FINAL_MERGED_ENCODED, index=False)
    print(f'已保存为 {FINAL_MERGED_ENCODED}')
    print(f'最终数据: {len(final)} 条记录')


if __name__ == '__main__':
    encode_merged()
