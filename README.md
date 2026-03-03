# 音乐推荐系统

本项目实现了基于内容推荐、协同过滤（UserCF、ItemCF、SVD）和深度学习（NCF）的音乐推荐系统，配有Streamlit前端可视化界面。

## 目录结构

```
项目根目录/
├── app/                    # 应用主目录
│   ├── main.py            # 主入口文件
│   ├── pages/             # 页面模块
│   │   ├── cold_start.py          # 冷启动推荐页面
│   │   ├── collaborative.py       # 协同过滤推荐页面
│   │   ├── deep_learning.py        # 深度学习推荐页面
│   │   └── content_based.py       # 内容推荐页面
│   └── utils/             # 工具模块
│       ├── helpers.py              # 工具函数
│       └── ui_components.py        # UI组件
├── src/                   # 推荐算法源码
│   ├── recommend_utils.py         # 推荐算法工具函数
│   └── deep_learning_recommend.py # NCF模型定义和训练
├── data_processing/       # 数据处理脚本
│   ├── preprocess_triplets.py
│   ├── merge_and_clean.py
│   └── ...
├── data/                  # 数据目录
│   ├── track_metadata.db
│   └── train_triplets.txt
├── model/                 # 模型目录
│   └── ncf_model.pth
├── config.py              # 配置文件
├── requirements.txt       # 依赖文件
└── README.md
```

##  主要功能

- **热门推荐（冷启动）**：为新用户推荐全局最热门的歌曲
- **协同过滤推荐**：支持UserCF、ItemCF、SVD三种算法，并基于当前登录用户生成个性化结果
- **深度学习推荐**：使用NCF（神经协同过滤）模型
- **内容推荐**：基于用户历史行为和内容特征的个性化推荐
- **用户历史管理**：支持勾选保存、实时更新历史记录
- **数据可视化**：词云图展示最受欢迎的歌手和歌曲

##  使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

- 将原始数据文件放入 `data/` 目录：
  - `track_metadata.db` - 歌曲元数据
  - `train_triplets.txt` - 用户行为数据
- 运行数据处理脚本生成 `final_merged_encoded_usernorm.csv`

### 3. 训练模型（可选）

如果需要使用深度学习推荐功能，需要先训练NCF模型：

```bash
python src/deep_learning_recommend.py
```

训练完成后，模型会保存到 `model/ncf_model.pth`

### 4. 运行前端

```bash
streamlit run app/main.py
```

### 5. 访问系统

浏览器访问：http://localhost:8501

##  主要依赖

- Python 3.7+
- Streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- wordcloud >= 1.9.0
- torch >= 2.0.0
- surprise >= 0.1
- requests >= 2.28.0

##  配置说明

所有配置都在 `config.py` 文件中，包括：
- 数据文件路径
- 模型文件路径
- 采样数量
- 推荐数量范围
- 用户历史记录大小等

##  数据集

- **track_metadata.db**：包含歌曲元数据（歌手、歌名、年份、熟悉度、热度等）
- **train_triplets.txt**：包含用户-歌曲-播放量三元组

数据集获取：
- 百度网盘：[点击下载](https://pan.baidu.com/s/1V1_Uvcx9Tj06feR7NnFqwg#list/path=%2F)  
- 提取码：rlnj

##  功能特点

1. **多算法融合**：集成4种推荐算法，适应不同场景
2. **用户体验优化**：历史记录管理、实时更新、加载提示
3. **性能优化**：数据缓存、模型缓存、智能采样
4. **代码结构清晰**：模块化设计，易于维护和扩展
5. **错误处理完善**：友好的错误提示和异常处理

##  开发说明

### 代码组织

- `app/` - 应用主目录，包含所有页面和UI组件
- `src/` - 推荐算法源码
- `data_processing/` - 数据处理脚本
- `config.py` - 统一配置文件

### 添加新功能

1. 新页面：在 `app/pages/` 目录下创建新文件
2. 新算法：在 `src/` 目录下添加算法实现
3. 新工具函数：在 `app/utils/helpers.py` 中添加

##  贡献

欢迎提交issue和pull request

##  License

仅供学习交流，禁止商用
