"""
对播放量进行用户归一化处理
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import FINAL_MERGED_ENCODED, DATA_FILE


def usernorm_rating():
    """对播放量进行用户归一化处理"""
    print('正在读取final_merged_encoded.csv...')
    df = pd.read_csv(FINAL_MERGED_ENCODED)
    print(f'读取到 {len(df)} 条记录')
    
    # 计算每个用户的最大点击量
    print('正在计算用户最大播放量...')
    user_max = df.groupby('user')['play_count'].transform('max')
    
    # 归一化比值
    print('正在计算归一化比值...')
    r = df['play_count'] / user_max
    
    # 评分（使用对数变换）
    print('正在计算归一化评分...')
    df['rating'] = np.log(2 + r)
    
    # 用归一化评分替换play_count
    final = df.copy()
    final['play_count'] = final['rating']
    final = final.drop(columns=['rating'])
    
    # 保存新数据
    final.to_csv(DATA_FILE, index=False)
    print(f'已保存归一化评分数据为 {DATA_FILE}')
    print(f'最终数据: {len(final)} 条记录')
    print(f'用户数量: {final["user"].nunique()}, 歌曲数量: {final["song"].nunique()}')


if __name__ == '__main__':
    usernorm_rating()
