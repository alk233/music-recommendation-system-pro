import csv
import os

from config import DATA_FILE


def count_csv_rows(path: str) -> int:
    # 流式计数，避免一次性加载大文件占内存
    with open(path, "r", encoding="utf-8", newline="") as f:
        # 减去表头 1 行
        return max(sum(1 for _ in f) - 1, 0)


def read_header(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def main():
    print(f"数据文件路径: {DATA_FILE}")
    if not os.path.isfile(DATA_FILE):
        print("文件不存在，请检查 config.py 中 DATA_FILE 配置。")
        return

    rows = count_csv_rows(DATA_FILE)
    header = read_header(DATA_FILE)
    print(f"总行数(不含表头): {rows}")
    print(f"字段数: {len(header)}")
    print(f"字段名: {header}")


if __name__ == "__main__":
    main()
