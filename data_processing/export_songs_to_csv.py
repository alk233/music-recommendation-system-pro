"""
导出歌曲元数据为CSV文件
"""
import sys
import os
import sqlite3
import pandas as pd

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import METADATA_DB, SONGS_CSV


def export_songs():
    """从数据库导出歌曲信息为CSV"""
    print('正在连接数据库...')
    conn = sqlite3.connect(METADATA_DB)
    
    print('正在读取songs表...')
    songs = pd.read_sql_query('SELECT * FROM songs', conn)
    
    print('正在保存为CSV文件...')
    songs.to_csv(SONGS_CSV, index=False, encoding='utf-8')
    
    conn.close()
    print(f'已成功将songs表导出为 {SONGS_CSV}')
    print(f'共导出 {len(songs)} 条记录')


if __name__ == '__main__':
    export_songs()
