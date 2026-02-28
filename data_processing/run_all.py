"""
数据处理主脚本 - 按顺序运行所有数据处理步骤
"""
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_processing.export_songs_to_csv import export_songs
from data_processing.filter_triplets import filter_triplets
from data_processing.merge_filtered_with_songs import merge_filtered_with_songs
from data_processing.encode_merged import encode_merged
from data_processing.usernorm_rating import usernorm_rating


def run_all():
    """按顺序运行所有数据处理步骤"""
    print("=" * 60)
    print("开始数据处理流程")
    print("=" * 60)
    
    steps = [
        ("步骤1: 导出歌曲元数据", export_songs),
        ("步骤2: 过滤活跃用户和热门歌曲", filter_triplets),
        ("步骤3: 合并过滤后的行为数据和歌曲元数据", merge_filtered_with_songs),
        ("步骤4: 对数据进行编码", encode_merged),
        ("步骤5: 用户归一化评分", usernorm_rating),
    ]
    
    for step_name, step_func in steps:
        print("\n" + "=" * 60)
        print(step_name)
        print("=" * 60)
        try:
            step_func()
            print(f"✓ {step_name} 完成")
        except Exception as e:
            print(f"✗ {step_name} 失败: {str(e)}")
            print("是否继续执行下一步？(y/n): ", end="")
            choice = input().strip().lower()
            if choice != 'y':
                print("数据处理流程已中断")
                return
    
    print("\n" + "=" * 60)
    print("数据处理流程全部完成！")
    print("=" * 60)
    print(f"最终数据文件: final_merged_encoded_usernorm.csv")


if __name__ == '__main__':
    run_all()
