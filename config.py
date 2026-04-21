# 配置文件
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件路径
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 主数据表：须含 user, song, play_count 及内容推荐所需列（与 data_processing 产出一致）
# 默认放在项目根目录（与本 config.py 同级）。若误放在 data/ 下，会自动选用存在的那一个。
_DATA_PRIMARY = os.path.join(BASE_DIR, 'final_merged_encoded_usernorm.csv')
_DATA_IN_DATA_DIR = os.path.join(DATA_DIR, 'final_merged_encoded_usernorm.csv')
if os.path.isfile(_DATA_PRIMARY):
    DATA_FILE = _DATA_PRIMARY
elif os.path.isfile(_DATA_IN_DATA_DIR):
    DATA_FILE = _DATA_IN_DATA_DIR
else:
    DATA_FILE = _DATA_PRIMARY
TRIPLETS_FILE = os.path.join(DATA_DIR, 'train_triplets.txt')
METADATA_DB = os.path.join(DATA_DIR, 'track_metadata.db')

# 数据处理中间文件路径
FILTERED_TRIPLETS = os.path.join(BASE_DIR, 'filtered_triplets.csv')
SONGS_CSV = os.path.join(BASE_DIR, 'songs.csv')
MERGED_CLEANED = os.path.join(BASE_DIR, 'merged_cleaned.csv')
FILTERED_MERGED_CLEANED = os.path.join(BASE_DIR, 'filtered_merged_cleaned.csv')
FINAL_MERGED_ENCODED = os.path.join(BASE_DIR, 'final_merged_encoded.csv')
TRIPLETS_PROCESSED = os.path.join(BASE_DIR, 'train_triplets_2M_processed.csv')

# 数据处理配置
TRIPLETS_SAMPLE_SIZE = 1_000_000  # 读取三元组数据量
TRIPLETS_PREPROCESS_SIZE = 2_000_000  # 预处理数据量
USER_MIN_PLAYS = 100  # 用户最小播放量
SONG_MIN_PLAYS = 50  # 歌曲最小播放量

# 模型文件路径
MODEL_DIR = os.path.join(BASE_DIR, 'model')
NCF_MODEL_PATH = os.path.join(MODEL_DIR, 'ncf_model.pth')
NCF_PERSONALIZED_MODEL_PATH = os.path.join(MODEL_DIR, 'ncf_model_personalized.pth')
CF_MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, 'cf_models.pkl')
CF_USERCF_MODEL_PATH = os.path.join(MODEL_DIR, 'usercf_model.pkl')
CF_ITEMCF_MODEL_PATH = os.path.join(MODEL_DIR, 'itemcf_model.pkl')
CF_SVD_MODEL_PATH = os.path.join(MODEL_DIR, 'svd_model.pkl')
CF_META_PATH = os.path.join(MODEL_DIR, 'cf_meta.json')
CONTENT_MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, 'content_encoder.pkl')

# 字体文件路径
FONT_PATH = os.path.join(BASE_DIR, 'simhei.ttf')
FONT_URL = "https://github.com/owent-utils/font/raw/master/simhei.ttf"

# 推荐算法配置
CF_SAMPLE_SIZE = 100000  # 协同过滤采样数量
# KNN 邻居数，越小越快（略降精度）；Surprise 默认约 40
CF_KNN_K = 28
# 协同过滤预测分下限（Surprise 估计分，约 [1,5]）；低于该阈值的候选不进入推荐列表
CF_MIN_EST_SCORE = 2.0
NCF_SAMPLE_SIZE = 1000000  # 深度学习训练读取行数上限
CONTENT_SAMPLE_SIZE = 100000  # 内容推荐采样数量（歌曲特征表构建时的交互行数上限）

# Streamlit 可视化性能：限制单次读 CSV 的最大行数，避免冷启动/分析页/歌曲信息全表扫描卡死
# 需要更贴近全量统计时可调大（会更慢、更占内存）
STREAMLIT_DATA_NROWS = 100000

# 推荐数量配置
MIN_RECOMMEND = 1
MAX_RECOMMEND = 20
DEFAULT_RECOMMEND = 10

# 用户历史记录配置
MAX_HISTORY_SIZE = 50  # 最大历史条数
HISTORY_PAGE_SIZE = 10  # 每页条数
HISTORY_MAX_PAGES = 5  # 最多页数
HISTORY_FILE = os.path.join(BASE_DIR, 'user_history.json')  # 历史记录文件路径

# 用户注册/登录配置
USERS_FILE = os.path.join(BASE_DIR, 'users.json')  # 用户注册信息文件路径

# Streamlit配置
PAGE_TITLE = "音乐推荐系统"
PAGE_LAYOUT = "wide"
