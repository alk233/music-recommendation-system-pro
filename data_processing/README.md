# 数据处理脚本说明

本目录包含所有数据处理相关的脚本，用于从原始数据生成最终可用的数据文件。

## 📋 脚本说明

### 1. export_songs_to_csv.py
- **功能**：从数据库导出歌曲元数据为CSV文件
- **输入**：`data/track_metadata.db`
- **输出**：`songs.csv`

### 2. filter_triplets.py
- **功能**：过滤活跃用户和热门歌曲
- **输入**：`data/train_triplets.txt`
- **输出**：`filtered_triplets.csv`

### 3. merge_filtered_with_songs.py
- **功能**：合并过滤后的行为数据和歌曲元数据
- **输入**：`filtered_triplets.csv`, `songs.csv`
- **输出**：`filtered_merged_cleaned.csv`

### 4. encode_merged.py
- **功能**：对合并后的数据进行Label编码
- **输入**：`filtered_triplets.csv`, `songs.csv`
- **输出**：`final_merged_encoded.csv`

### 5. usernorm_rating.py
- **功能**：对播放量进行用户归一化处理
- **输入**：`final_merged_encoded.csv`
- **输出**：`final_merged_encoded_usernorm.csv`（最终数据文件）

### 6. run_all.py ⭐
- **功能**：主脚本，按顺序运行所有数据处理步骤
- **使用**：`python data_processing/run_all.py`

## 🔄 使用方式

### 方式1：使用主脚本（推荐）
```bash
python data_processing/run_all.py
```

### 方式2：手动按顺序运行
```bash
python data_processing/export_songs_to_csv.py
python data_processing/filter_triplets.py
python data_processing/merge_filtered_with_songs.py
python data_processing/encode_merged.py
python data_processing/usernorm_rating.py
```

## ⚙️ 配置说明

所有路径和参数都在 `config.py` 中配置，无需手动修改脚本。
