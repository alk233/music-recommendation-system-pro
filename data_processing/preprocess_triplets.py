"""
预处理三元组数据（Label编码）
"""
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import TRIPLETS_FILE, TRIPLETS_PROCESSED, TRIPLETS_PREPROCESS_SIZE


def preprocess_triplets():
    """对三元组数据进行Label编码"""
    print(f'正在读取前{TRIPLETS_PREPROCESS_SIZE}条数据...')
    df = pd.read_csv(
        TRIPLETS_FILE,
        sep='\t',
        names=['user', 'song', 'play_count'],
        nrows=TRIPLETS_PREPROCESS_SIZE,
        dtype={'user': 'object', 'song': 'object', 'play_count': 'int32'}
    )
    print(f'读取到 {len(df)} 条记录')
    
    # 对user和song进行Label编码
    print('正在进行Label编码...')
    user_encoder = LabelEncoder()
    song_encoder = LabelEncoder()
    
    df['user'] = user_encoder.fit_transform(df['user'])
    df['song'] = song_encoder.fit_transform(df['song'])
    
    print(f'用户数量: {df["user"].nunique()}, 歌曲数量: {df["song"].nunique()}')
    
    # 保存处理后的数据
    print('正在保存处理后的数据...')
    df.to_csv(TRIPLETS_PROCESSED, index=False)
    print(f'处理完成，已保存为 {TRIPLETS_PROCESSED}')


if __name__ == '__main__':
    preprocess_triplets()
