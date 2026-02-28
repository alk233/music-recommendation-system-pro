"""
合并过滤后的行为数据和歌曲元数据
"""
import sys
import os
import pandas as pd

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import FILTERED_TRIPLETS, SONGS_CSV, FILTERED_MERGED_CLEANED


def merge_filtered_with_songs():
    """合并过滤后的行为数据和歌曲元数据"""
    # 1. 读取过滤后的行为数据和歌曲元数据
    print('正在读取filtered_triplets.csv...')
    df = pd.read_csv(FILTERED_TRIPLETS)
    print(f'读取到 {len(df)} 条过滤后的行为数据')
    
    print('正在读取songs.csv...')
    songs = pd.read_csv(SONGS_CSV)
    print(f'读取到 {len(songs)} 条歌曲元数据')
    
    # 2. 合并（用song_id作为键）
    print('正在合并数据...')
    merged = pd.merge(df, songs, left_on='song_id', right_on='song_id', how='left')
    print(f'合并后数据: {len(merged)} 条记录')
    
    # 3. 删除无用字段
    print('正在删除无用字段...')
    drop_cols = [
        'track_id', 'artist_id', 'artist_mbid', 'duration',
        'track_7digitalid', 'shs_perf', 'shs_work'
    ]
    merged_cleaned = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    
    # 4. 去重
    print('正在去重...')
    before_dedup = len(merged_cleaned)
    merged_cleaned = merged_cleaned.drop_duplicates()
    after_dedup = len(merged_cleaned)
    print(f'去重前: {before_dedup} 条，去重后: {after_dedup} 条')
    
    # 5. 保存结果
    merged_cleaned.to_csv(FILTERED_MERGED_CLEANED, index=False)
    print(f'合并并清洗后的数据已保存为 {FILTERED_MERGED_CLEANED}')


if __name__ == '__main__':
    merge_filtered_with_songs()
