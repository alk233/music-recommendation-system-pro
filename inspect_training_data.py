"""
快速查看训练数据或原始数据概况。

用法（在项目根目录执行）：
  python inspect_training_data.py
  python inspect_training_data.py --source processed
  python inspect_training_data.py --source raw_triplets --nrows 10
  python inspect_training_data.py --source raw_metadata --nrows 10
"""

from __future__ import annotations

import argparse
import os
import sys
import sqlite3

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_FILE, TRIPLETS_FILE, METADATA_DB  # noqa: E402


def fmt_num(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def inspect_dataset(nrows: int | None, sample: int, user_id: int | None = None) -> None:
    print("=" * 72)
    print("训练数据集检查")
    print("=" * 72)
    print(f"数据文件: {DATA_FILE}")
    if not os.path.isfile(DATA_FILE):
        print("错误: 数据文件不存在。")
        return

    usecols = ["user", "song", "play_count", "artist_name", "title", "year"]
    print(f"读取行数: {'全量' if nrows is None else nrows}")
    df = pd.read_csv(DATA_FILE, nrows=nrows)
    print(f"实际读取: {fmt_num(len(df))} 行")
    print(f"字段数量: {len(df.columns)}")
    print(f"字段列表: {list(df.columns)}")
    print("-" * 72)

    # 关键字段检查
    required = ["user", "song", "play_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"缺少关键字段: {missing}")
        return

    print("关键字段空值:")
    for c in required:
        na = int(df[c].isna().sum())
        print(f"  - {c}: {fmt_num(na)}")
    print("-" * 72)

    # 基础统计
    n_users = df["user"].nunique(dropna=True)
    n_songs = df["song"].nunique(dropna=True)
    n_interactions = len(df)
    density = (n_interactions / (max(1, n_users) * max(1, n_songs))) * 100

    print("交互统计:")
    print(f"  - 交互数: {fmt_num(n_interactions)}")
    print(f"  - 用户数: {fmt_num(n_users)}")
    print(f"  - 歌曲数: {fmt_num(n_songs)}")
    print(f"  - 稀疏度(密度%): {density:.6f}%")
    print("-" * 72)

    # play_count 分布
    s = pd.to_numeric(df["play_count"], errors="coerce")
    print("play_count 分布:")
    print(f"  - min/max: {s.min():.6f} / {s.max():.6f}")
    print(f"  - mean/std: {s.mean():.6f} / {s.std():.6f}")
    print(
        f"  - p50/p90/p99: {s.quantile(0.5):.6f} / "
        f"{s.quantile(0.9):.6f} / {s.quantile(0.99):.6f}"
    )
    print("-" * 72)

    # 用户与歌曲交互分布
    user_cnt = df.groupby("user").size()
    song_cnt = df.groupby("song").size()
    print("每用户交互分布:")
    print(
        f"  - min/median/p90/max: {user_cnt.min()} / "
        f"{user_cnt.median():.0f} / {user_cnt.quantile(0.9):.0f} / {user_cnt.max()}"
    )
    print("每歌曲交互分布:")
    print(
        f"  - min/median/p90/max: {song_cnt.min()} / "
        f"{song_cnt.median():.0f} / {song_cnt.quantile(0.9):.0f} / {song_cnt.max()}"
    )
    print("-" * 72)

    print("历史条数最多的用户（当前读取范围 Top10）:")
    top_users = user_cnt.sort_values(ascending=False).head(10)
    for uid, cnt in top_users.items():
        print(f"  - user={uid}: {cnt} 条")
    print("-" * 72)

    if user_id is not None:
        uid = int(user_id)
        user_rows = df[df["user"] == uid]
        print(f"指定用户历史条数查询: user={uid}")
        print(f"  - 历史条数: {len(user_rows)}")
        if len(user_rows) > 0:
            cols = [c for c in ["song", "title", "artist_name", "play_count"] if c in user_rows.columns]
            print(f"  - 该用户示例记录（前 {sample} 条）:")
            print(user_rows[cols].head(sample).to_string(index=False))
        print("-" * 72)

    # 示例行
    print(f"示例数据（前 {sample} 行）:")
    print(df.head(sample).to_string(index=False))
    print("-" * 72)

    # 热门歌曲示例（如果有名称列）
    if "artist_name" in df.columns and "title" in df.columns:
        print("Top10 热门歌曲（按交互次数）:")
        top = (
            df.groupby(["song", "artist_name", "title"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .head(10)
        )
        print(top.to_string(index=False))
        print("-" * 72)

    print("检查完成。")


def inspect_raw_triplets(nrows: int, sample: int) -> None:
    print("=" * 72)
    print("原始三元组检查（train_triplets.txt）")
    print("=" * 72)
    print(f"数据文件: {TRIPLETS_FILE}")
    if not os.path.isfile(TRIPLETS_FILE):
        print("错误: 原始三元组文件不存在。")
        return

    df = pd.read_csv(
        TRIPLETS_FILE,
        sep="\t",
        header=None,
        names=["user_raw", "song_raw", "play_count_raw"],
        nrows=nrows,
    )
    print(f"实际读取: {fmt_num(len(df))} 行")
    print(f"字段列表: {list(df.columns)}")
    print("-" * 72)
    print(f"示例数据（前 {sample} 行）:")
    print(df.head(sample).to_string(index=False))
    print("-" * 72)
    print("检查完成。")


def inspect_raw_metadata(nrows: int) -> None:
    print("=" * 72)
    print("原始元数据检查（track_metadata.db）")
    print("=" * 72)
    print(f"数据文件: {METADATA_DB}")
    if not os.path.isfile(METADATA_DB):
        print("错误: 原始元数据库文件不存在。")
        return

    conn = sqlite3.connect(METADATA_DB)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn,
        )["name"].tolist()
        if not tables:
            print("数据库中没有表。")
            return
        print(f"表列表: {tables}")
        table = tables[0]
        print(f"默认查看表: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {int(nrows)}", conn)
        print(f"字段列表: {list(df.columns)}")
        print("-" * 72)
        print(f"示例数据（前 {int(nrows)} 行）:")
        print(df.to_string(index=False))
        print("-" * 72)
        print("检查完成。")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="检查训练数据或原始数据概况")
    parser.add_argument(
        "--source",
        type=str,
        default="processed",
        choices=["processed", "raw_triplets", "raw_metadata"],
        help="数据源：processed(处理后主表) / raw_triplets(原始三元组) / raw_metadata(原始元数据库)",
    )
    parser.add_argument("--nrows", type=int, default=10, help="读取前 n 行（默认 10 行）")
    parser.add_argument("--sample", type=int, default=10, help="打印样例行数")
    parser.add_argument("--user-id", type=int, default=None, help="指定用户ID，查询该用户历史条数")
    args = parser.parse_args()
    if args.source == "processed":
        inspect_dataset(args.nrows, args.sample, user_id=args.user_id)
    elif args.source == "raw_triplets":
        inspect_raw_triplets(args.nrows, args.sample)
    else:
        inspect_raw_metadata(args.nrows)


if __name__ == "__main__":
    main()

