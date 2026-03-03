# 配置文件
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(BASE_DIR, 'final_merged_encoded_usernorm.csv')
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

# 字体文件路径
FONT_PATH = os.path.join(BASE_DIR, 'simhei.ttf')
FONT_URL = "https://github.com/owent-utils/font/raw/master/simhei.ttf"

# 推荐算法配置
CF_SAMPLE_SIZE = 50000  # 协同过滤采样数量
NCF_SAMPLE_SIZE = 50000  # 深度学习采样数量
CONTENT_SAMPLE_SIZE = 10000  # 内容推荐采样数量

# 推荐数量配置
MIN_RECOMMEND = 1
MAX_RECOMMEND = 20
DEFAULT_RECOMMEND = 10

# 用户历史记录配置
MAX_HISTORY_SIZE = 10  # 最大历史记录数量
HISTORY_FILE = os.path.join(BASE_DIR, 'user_history.json')  # 历史记录文件路径

# 用户注册/登录配置
USERS_FILE = os.path.join(BASE_DIR, 'users.json')  # 用户注册信息文件路径

# Streamlit配置
PAGE_TITLE = "音乐推荐系统"
PAGE_LAYOUT = "wide"
