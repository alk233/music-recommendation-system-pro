# 音乐推荐系统（Multi-Algorithm Music Recommender）

一个推荐系统实验的完整项目：  
从原始 `triplets + metadata` 数据处理开始，覆盖 `UserCF / ItemCF / SVD / 内容推荐 / NCF / 融合推荐`，并提供可交互的 Streamlit 前端与离线评测页面。

---

## 目录

- [1. 项目目标](#1-项目目标)
- [2. 功能总览](#2-功能总览)
- [3. 技术栈与核心依赖](#3-技术栈与核心依赖)
- [4. 项目结构](#4-项目结构)
- [5. 快速开始](#5-快速开始)
- [6. 数据处理全流程](#6-数据处理全流程)
- [7. 模型训练与持久化机制](#7-模型训练与持久化机制)
- [8. 前端页面说明](#8-前端页面说明)
- [9. 离线评测协议与指标解释](#9-离线评测协议与指标解释)
- [10. 关键配置项](#10-关键配置项)
- [11. 实验建议与最佳实践](#11-实验建议与最佳实践)
- [12. 功能迭代与设计决策（改动+目的）](#12-功能迭代与设计决策改动目的)
- [13. 许可与用途](#13-许可与用途)

---

## 1. 项目目标

本项目解决的问题是：  
**基于用户历史听歌行为，为当前用户推荐“可能喜欢”的歌曲，并支持不同算法对比、融合和可视化分析。**

项目强调三件事：

- **可运行**：一套代码可直接启动并展示推荐结果；
- **可解释**：每个页面给出指标说明和算法特性；
- **可实验**：支持单模型/融合模型调参与离线评测。

---

## 2. 功能总览

- 多算法推荐：`UserCF`、`ItemCF`、`SVD`、`内容推荐`、`NCF`
- 五路融合推荐：`UserCF + ItemCF + SVD + NCF + Content`
- 用户系统：注册/登录、用户历史持久化
- 推荐结果回放：推荐歌曲可一键加入历史记录
- 自动/手动重训：支持按模型范围重训并持久化
- 算法对比页：
  - 工程效果对比（读取已持久化模型）
  - 同规模公平对比（固定行数现场训练）
- 评测指标：`HR@K`、`Recall@K`、`NDCG@K`、`P@K`、`MRR`、`F1@K`、融合行 `候选AUC`

---

## 3. 技术栈与核心依赖

- **语言与框架**：Python、Streamlit
- **数据与数值计算**：Pandas、NumPy
- **经典推荐算法**：Surprise（KNNBasic / SVD）
- **深度学习**：PyTorch（NCF + BPR 训练）
- **特征工程**：scikit-learn（`FeatureHasher`、`MinMaxScaler`、`QuantileTransformer`）

依赖见 `requirements.txt`。

---

## 4. 项目结构

```text
music-recommendation-system-pro/
├── app/
│   ├── main.py                           # Streamlit 入口（路由、登录态、侧栏）
│   ├── pages/
│   │   ├── cold_start.py                 # 冷启动/热门榜单
│   │   ├── collaborative.py              # 协同过滤页（UserCF / ItemCF / SVD）
│   │   ├── deep_learning.py              # NCF 页面
│   │   ├── content_based.py              # 内容推荐页面
│   │   ├── hybrid.py                     # 融合推荐页面（五路权重）
│   │   ├── algorithm_compare.py          # 多算法离线评测对比
│   │   └── analysis.py                   # 用户画像与全站热门分析
│   └── utils/
│       ├── helpers.py                    # 重训逻辑、训练元数据、用户/历史读写
│       ├── ui_components.py              # 通用 UI 组件
│       ├── background_prefetch.py        # 空闲预取
│       └── data_notes.py                 # 页面指标说明文本
├── src/
│   ├── recommend_utils.py                # 主推荐逻辑（CF/NCF/Content/Hybrid）
│   └── deep_learning_recommend.py        # NCF 模型定义与训练
├── evaluation/
│   └── compare_algorithms.py             # 离线评测核心
├── data_processing/
│   ├── run_all.py                        # 数据处理一键流水线
│   ├── export_songs_to_csv.py            # DB -> songs.csv
│   ├── filter_triplets.py                # 用户/歌曲活跃度过滤
│   ├── merge_filtered_with_songs.py      # 行为与元数据合并
│   ├── encode_merged.py                  # user/song 编码
│   ├── usernorm_rating.py                # 标签归一化（最终主表）
│   └── README.md                         # 数据处理脚本说明
├── model/                                # 模型文件与元数据目录（运行后生成）
├── config.py                             # 路径、采样规模、算法配置
├── inspect_training_data.py              # 训练数据检查脚本
├── train_and_validate_models.py          # 训练+抽样验证脚本
├── count_dataset_rows.py                 # 数据行数统计脚本
├── user_history.json                     # 用户历史（运行后生成）
├── users.json                            # 注册用户信息（运行后生成）
└── README.md
```

---

## 5. 快速开始

### 5.1 环境准备

```bash
pip install -r requirements.txt
```

建议使用独立虚拟环境（conda / venv）。

### 5.2 数据准备

主训练文件默认为：

- `final_merged_encoded_usernorm.csv`（项目根目录）
- 或 `data/final_merged_encoded_usernorm.csv`

#### 数据集来源

- 百度网盘下载地址：<https://pan.baidu.com/s/1V1_Uvcx9Tj06feR7NnFqw#/list/path=%2F>
- 提取码：`rlnj`

> 说明：下载后请将原始数据按本仓库的数据处理流程转换为 `final_merged_encoded_usernorm.csv` 再启动系统。

若尚未生成，可执行：

```bash
python data_processing/run_all.py
```

### 5.3 启动系统

```bash
streamlit run app/main.py
```

默认访问：<http://localhost:8501>

### 5.4 可选：单独训练 NCF

```bash
python src/deep_learning_recommend.py
```

---

## 6. 数据处理全流程

这一部分是项目从“原始数据”到“训练主表”的完整流程。

### 6.1 输入数据

- 行为数据：`data/train_triplets.txt`
- 元数据：`data/track_metadata.db`

### 6.2 流程步骤

#### Step 1: 导出歌曲元数据

脚本：`data_processing/export_songs_to_csv.py`

- 输入：`track_metadata.db`
- 输出：`songs.csv`
- 目的：
  - 将 SQLite 元数据统一转成 CSV，便于后续脚本串联处理；
  - 让行为数据和歌曲属性数据进入同一种表结构，降低合并复杂度。

#### Step 2: 过滤行为日志

脚本：`data_processing/filter_triplets.py`

- 输入：`train_triplets.txt`
- 过滤规则（来自 `config.py`）：
  - `USER_MIN_PLAYS`（用户总播放量阈值）
  - `SONG_MIN_PLAYS`（歌曲总播放量阈值）
- 输出：`filtered_triplets.csv`
- 目的：
  - 去掉超低活跃用户与超冷门歌曲，减少噪声交互；
  - 提升后续协同模型（尤其 KNN 类）的可学习性与训练稳定性；
  - 在不损失主干信号的情况下控制数据规模与内存压力。

#### Step 3: 合并行为与元数据

脚本：`data_processing/merge_filtered_with_songs.py`

- 输入：`filtered_triplets.csv` + `songs.csv`
- 处理：
  - 按 `song_id` 左连接
  - 删除无用字段
  - 去重
- 输出：`filtered_merged_cleaned.csv`
- 目的：
  - 将“用户行为”与“歌曲内容属性”打通，为内容推荐与可视化准备统一数据底座；
  - 去除无关字段与重复行，降低后续编码与训练成本。

#### Step 4: 编码 user/song ID

脚本：`data_processing/encode_merged.py`

- 对 `user_id`、`song_id` 做 `LabelEncoder`
- 输出字段包含：
  - `user`、`song`、`play_count`
  - `title`、`release`、`artist_name`
  - `artist_familiarity`、`artist_hotttnesss`、`year`
- 输出：`final_merged_encoded.csv`
- 目的：
  - 将原始离散 ID 映射为连续整数索引，满足协同过滤与嵌入模型输入要求；
  - 统一主键空间，避免后续训练/推理阶段出现类型与索引不一致问题。

#### Step 5: 标签归一化（核心）

脚本：`data_processing/usernorm_rating.py`

对 `play_count` 做以下变换：

1. 用户内归一化：`r = play_count / user_max_play_count`
2. 平滑：`t = ln(2 + r)`
3. 全局分位数映射（`QuantileTransformer`）到近似均匀 `[0,1]`

最终输出：`final_merged_encoded_usernorm.csv`
- 目的：
  - 缓解不同用户“播放量口径不一致”带来的偏差（重度用户与轻度用户可比较）；
  - 压缩长尾极值，减少少量超大播放次数对模型学习的主导效应；
  - 得到稳定的偏好强度标签，便于 CF、SVD、内容推荐和可视化统一使用。

### 6.3 一键执行

```bash
python data_processing/run_all.py
```

---

## 7. 模型训练与持久化机制

### 7.1 全局模型文件

`model/` 下主要文件：

- `usercf_model.pkl`  
  - 含义：UserCF 持久化模型文件。  
  - 作用：保存用户协同过滤训练结果，避免每次启动重复训练。用于协同过滤页/融合页/工程对比。

- `itemcf_model.pkl`  
  - 含义：ItemCF 持久化模型文件。  
  - 作用：保存物品协同过滤训练结果。由于 ItemCF 内存敏感，单独持久化便于独立重训和快速回滚。

- `svd_model.pkl`  
  - 含义：SVD 持久化模型文件。  
  - 作用：保存矩阵分解模型参数，供协同过滤页、融合页和工程对比直接加载。

- `cf_meta.json`  
  - 含义：CF 三路模型元数据。  
  - 作用：记录三路训练行数、SVD 参数、截断策略、样本规模（rows/users/songs）等，用于判断是否需要重训以及页面展示“当前模型口径”。

- `content_encoder.pkl`  
  - 含义：内容推荐特征编码器与歌曲特征缓存。  
  - 作用：保存 `FeatureHasher/Scaler/songs` 等对象，避免每次重建内容特征矩阵，提升内容推荐页面与融合推荐的响应速度。

- `ncf_model.pth`  
  - 含义：全局 NCF 模型文件。  
  - 作用：保存 NCF 主模型参数及索引映射（`user2idx/song2idx`），用于全局推荐和算法对比中的工程模式评测。

- `ncf_model_personalized.pth`（个性化 NCF）  
  - 含义：按当前用户历史微调得到的个性化 NCF 模型。  
  - 作用：与全局 NCF 分离保存，防止个性化训练覆盖全局模型，保证对比评测和线上全局推荐口径稳定。

- `training_meta.json`（训练参数/时间戳等）  
  - 含义：训练状态与参数回填元数据。  
  - 作用：记录上次重训时间、各模型最近训练参数、页面参数回填基线等，使前端能正确显示“上次训练值”和重训状态。

### 7.2 协同过滤（CF）机制

实现位置：`src/recommend_utils.py`

- CF 三路独立文件保存（UserCF / ItemCF / SVD）
- 协同过滤页支持按当前选中模型单独训练：
  - 选 `UserCF` -> 只重训 UserCF
  - 选 `ItemCF` -> 只重训 ItemCF
  - 选 `SVD` -> 只重训 SVD
- 当前策略下，`ItemCF` 使用实体截断（9000×9000）来控制内存规模；`UserCF` / `SVD` 不做同样截断。

### 7.3 NCF 机制

实现位置：`src/deep_learning_recommend.py`

- 模型：用户嵌入 + 歌曲嵌入 + MLP
- 训练目标：BPR 风格 pairwise loss
- 负采样：随机负样本 + 热门负样本混合（可调 `neg_k` 与 `popular_negative_ratio`）
- 训练增强：AMP、梯度裁剪、早停、学习率调度
- 持久化：保存 `state_dict + user2idx + song2idx + num_users + num_songs`

### 7.4 内容推荐机制

实现位置：`src/recommend_utils.py`

- 使用歌曲内容特征构建向量：
  - `artist_name`/`title`/`release` 的哈希特征
  - `year`/`artist_familiarity`/`artist_hotttnesss` 数值特征
- 用户画像：历史歌曲向量均值
- 候选打分：与用户画像余弦相似 + 轻量语义加成（如歌手/年份亲和）

### 7.5 融合推荐机制

实现位置：`src/recommend_utils.py` 的 `hybrid_recommend`

- 汇总五路候选分（UserCF/ItemCF/SVD/NCF/Content）
- 每路先做鲁棒校准（median/IQR + sigmoid）
- 再按权重加权，得到融合排序分

---

## 8. 前端页面说明

### 8.1 全局前端功能（跨页面）

- 用户注册/登录与会话管理（支持登出）
- 用户历史记录持久化（`user_history.json`），支持侧边栏分页浏览与删除
- 推荐结果一键“听歌”并写入历史，形成在线反馈闭环
- 各页“数据变更标记”与“上次重训时间”提示，帮助判断是否需要重训
- 指标说明折叠面板（统一解释各页分数来源与口径）
- 空闲预取（idle prefetch）机制，减少常见页面切换等待

### 首页（热门推荐）

- 冷启动热门榜单（无个性化历史也可使用）
- 热门歌曲/热门歌手展示与刷新
- 多维热门结果展示（用于快速预览样本分布）

### 排行榜（协同过滤）

- 模型切换：`UserCF` / `ItemCF` / `SVD`
- 仅显示当前模型训练行数输入框
- 强制重训按钮只重训当前模型
- 获取个性化推荐时只使用当前模型
- 显示当前模型训练内存预估（尤其 UserCF/ItemCF）
- 参数回填遵循“上次成功训练值”，避免临时编辑误持久化

### 深度学习推荐

- NCF 专属训练参数可调：
  - `train_rows`、`epochs`
  - `lr`、`emb_dim`、`neg_k`
  - `popular_negative_ratio`、`min_epochs`、`patience`、`batch_size`
- 支持手动触发 NCF 重训与个性化推荐查看

### 内容推荐

- 内容模型单独重训与推荐
- 基于用户历史与歌曲内容特征生成结果

### 融合推荐

- 可设置五路训练条数
- 可使用自定义五路权重
- 展示融合排序结果
- 可对 NCF 高级参数进行联动训练配置

### 算法对比评测

- 两种模式：
  - 工程效果对比（读取持久化模型）
  - 同规模公平对比（统一行数现场训练）
- 输出主算法表 + 基线表 + 指标图
- 指标显示格式统一（百分数可读化）并支持融合权重可视化

### 个人中心

- 用户画像（歌手偏好、年份分布等）
- 全站热门统计

---

## 9. 离线评测协议与指标解释

实现位置：`evaluation/compare_algorithms.py`

### 9.1 协议

- 每个测试样本：`1 正例 + 99 负例`
- 每个算法对候选集打分排序
- 评测指标统一计算，便于横向比较

### 9.2 指标

- `HR@K`：命中率
- `Recall@K`：当前单正例设定下与 HR@K 数值一致
- `NDCG@K`：排序质量
- `P@K`：准确率
- `MRR`：首个相关项的倒数排名
- `F1@K`：P/R 调和均值
- 融合行额外：`候选AUC`

> 重要：评测分数越高不一定代表“探索能力更强”。  
> 高相关性模型可能牺牲新颖性与多样性，这在推荐系统中是常见权衡。

---

## 10. 关键配置项

所有关键参数都在 `config.py`。

### 数据与路径

- `DATA_FILE`：主训练数据路径
- `MODEL_DIR`：模型输出目录
- `USERS_FILE`、`HISTORY_FILE`：用户与历史 JSON

### 数据处理阈值

- `USER_MIN_PLAYS`
- `SONG_MIN_PLAYS`

### 训练规模

- `CF_SAMPLE_SIZE`
- `CONTENT_SAMPLE_SIZE`
- `NCF_SAMPLE_SIZE`

### 推荐与过滤

- `CF_KNN_K`
- `CF_MIN_EST_SCORE`
- `MIN_RECOMMEND` / `MAX_RECOMMEND` / `DEFAULT_RECOMMEND`

---

## 11. 实验建议与最佳实践

### 12.1 基础实验流程（建议）

1. 先跑数据检查：

```bash
python inspect_training_data.py --source processed --nrows 100000
```

2. 先用中小规模参数跑通全流程（例如 10w）
3. 再逐步增加训练规模并记录指标变化
4. 对比时固定随机种子，至少跑 2~3 次做均值

### 12.2 融合调参建议

- 先用默认权重作为基线
- 每次只改 1~2 路权重，观察 `NDCG/MRR/AUC`
- 若某一路覆盖率或稳定性差，先降权再观察

### 12.3 报告撰写建议

建议在实验报告里同时给出：

- 准确性指标（HR/NDCG/MRR/F1）
- 模型训练规模（rows/users/songs）
- 资源成本（训练耗时、内存）
- 可解释性结论（为什么融合优于单模型）

---

## 12. 功能迭代与设计决策（改动+目的）

这一节用于说明“每个功能为什么这么改”，避免后来维护者只看到结果，不知道背景和目的。

### 13.1 NCF 相关

- **改动：训练时做 `user/song` 密集编码并随模型持久化映射表。**  
  **目的：** 解决稀疏原始 ID 直接作为 embedding 索引时大量参数未训练的问题，提升 NCF 实际可用性与稳定性。

- **改动：NCF 模型文件保存 `state_dict + user2idx + song2idx + num_users + num_songs`。**  
  **目的：** 保证推理和评测阶段可以正确回放训练时的索引空间，避免重启后映射丢失导致分数异常或空结果。

- **改动：前端恢复并细化 NCF 可调参数（`lr/emb_dim/neg_k/pop_ratio/min_epochs/patience/batch_size`）。**  
  **目的：** 让实验可控，支持在同一套代码内做系统化调参与复现实验。

- **改动：全局 NCF 与个性化 NCF 模型文件分离（`ncf_model.pth` 与 `ncf_model_personalized.pth`）。**  
  **目的：** 避免个性化训练覆盖全局模型，保证算法对比页和线上推荐口径稳定。

- **改动：公平对比模式下 NCF 改为“仅内存现场训练”。**  
  **目的：** 与同规模模式的“现场训练”语义一致，同时避免覆盖全局模型。

### 13.2 协同过滤（UserCF / ItemCF / SVD）相关

- **改动：三路模型文件独立保存（`usercf_model.pkl`、`itemcf_model.pkl`、`svd_model.pkl`）。**  
  **目的：** 支持按子模型独立重训、独立回滚，减少相互影响。

- **改动：协同过滤页“当前选中模型 -> 仅训练该模型”。**  
  **目的：** 让页面交互语义和后端训练语义一致，避免“选了 UserCF 却把三路全训”的误解。

- **改动：协同过滤页仅显示当前模型训练条数输入框。**  
  **目的：** 降低误操作，明确“当前页面只调整当前模型”。

- **改动：参数回填改为“上次训练成功值”，不把未训练的临时编辑值持久化。**  
  **目的：** 避免切换模型时参数串值，保证配置可追踪。

- **改动：个性化推荐阶段按当前算法单路训练，不再一次性训练三路。**  
  **目的：** 修复内存异常放大（尤其 ItemCF）与“训练条数不生效”问题。

- **改动：ItemCF 恢复 9000×9000 实体截断。**  
  **目的：** 控制物品-物品相似度矩阵内存峰值，避免在大规模物品集合下 OOM。

- **改动：SVD 参数与评分映射做过多轮调优并回稳。**  
  **目的：** 在你当前数据分布下兼顾排序指标与稳定性，避免激进参数带来的过拟合和波动。

### 13.3 内容推荐相关

- **改动：内容特征扩展到 `artist/title/release/year/familiarity/hotttnesss`，并做哈希+数值编码。**  
  **目的：** 提升内容向量表达力，在不依赖协同信号时提升召回质量。

- **改动：内容推荐改为向量化打分并增加语义加成（歌手/年份亲和）。**  
  **目的：** 在保持可解释性的同时提高排序稳定性与运行效率。

### 13.4 融合推荐相关

- **改动：五路权重归一、旧版 `cf` 权重兼容映射、默认权重动态可视化。**  
  **目的：** 避免前后端权重口径不一致，提高可解释性与调参可控性。

- **改动：融合使用“各路分数校准后再加权”。**  
  **目的：** 缓解不同模型原始分数量纲不一致导致的融合偏置问题。

### 13.5 算法对比与评测相关

- **改动：算法对比页文案改为动态读取真实训练规模，不再写死。**  
  **目的：** 减少“前端说明与实际训练数据不一致”的误导。

- **改动：工程对比与同规模公平对比分离。**  
  **目的：** 同时满足“看当前上线效果”和“做公平算法比较”两类需求。

- **改动：评测表固定两位小数显示。**  
  **目的：** 降低“看起来被取整”造成的误判，提升可读性。

### 13.6 用户系统与可视化相关

- **改动：用户注册/登录 + 历史记录 JSON 持久化，并可并入训练数据。**  
  **目的：** 支持真实交互闭环，让推荐系统具备持续学习能力。

- **改动：个人中心和侧边栏历史展示支持分页与删除。**  
  **目的：** 提升可操作性，避免历史列表过长影响体验。

### 13.7 关键工程原则（本项目遵循）

- **原则 1：界面语义必须与后端行为一致。**  
- **原则 2：可复现优先，训练参数与结果要可追踪。**  
- **原则 3：算法评测口径必须明确区分工程效果与公平对比。**  
- **原则 4：在内存受限环境下，优先保证系统可运行，再追求极限指标。**

---

## 13. 许可与用途

本项目主要用于学习、教学与研究演示。  
如需商用，请先确认数据与依赖库许可条款。
