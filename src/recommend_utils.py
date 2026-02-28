# 推荐算法工具模块
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from surprise import Dataset, Reader, KNNBasic, SVD as SurpriseSVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_FILE, NCF_MODEL_PATH, CF_SAMPLE_SIZE, NCF_SAMPLE_SIZE
from src.deep_learning_recommend import NCF

# 内容推荐冷启动
@st.cache_data
def _get_cold_start_data():
    # 缓存冷启动数据
    df = pd.read_csv(DATA_FILE)
    # 统计每首歌的总播放量并排序
    song_stats = df.groupby(['song', 'artist_name', 'title'])['play_count'].sum().reset_index()
    song_stats = song_stats.sort_values('play_count', ascending=False)
    return song_stats

def content_cold_start(topk=10):
    # 内容冷启动推荐
    song_stats = _get_cold_start_data()
    top_songs = song_stats.head(topk)
    result = [
        f"{row['artist_name']} - {row['title']} (song_id={row['song']})，总播放量：{row['play_count']:.2f}"
        for _, row in top_songs.iterrows()
    ]
    return result

# 协同过滤 UserCF/ItemCF/SVD
_df_cf = pd.read_csv(DATA_FILE, nrows=CF_SAMPLE_SIZE)
_reader = Reader(rating_scale=(_df_cf['play_count'].min(), _df_cf['play_count'].max()))
_data = Dataset.load_from_df(_df_cf[['user', 'song', 'play_count']], _reader)
_trainset, _testset = train_test_split(_data, test_size=0.2, random_state=42)

# 预加载歌曲信息
_song_info = _df_cf.drop_duplicates('song')[['song', 'title', 'artist_name']]
_song_info_dict = _song_info.set_index('song').apply(lambda x: f"{x['artist_name']} - {x['title']}", axis=1).to_dict()

# UserCF
_sim_options_user = {'name': 'cosine', 'user_based': True}
_algo_usercf = KNNBasic(sim_options=_sim_options_user)
_algo_usercf.fit(_trainset)

@st.cache_resource
def get_usercf_model():
    # 获取UserCF模型
    model = KNNBasic(sim_options=_sim_options_user)
    model.fit(_trainset)
    return model

def usercf_topn(topk):
    # UserCF推荐TopN
    model = get_usercf_model()
    user_id = str(_df_cf['user'].iloc[0])
    all_songs = _df_cf['song'].unique()
    user_listened = _df_cf[_df_cf['user'] == int(user_id)]['song'].tolist()
    to_predict = [song for song in all_songs if song not in user_listened]
    recommendations = []
    for song in to_predict[:100]:
        pred = model.predict(user_id, song)
        recommendations.append((song, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    result = [f"{_song_info_dict.get(song, '未知')} (song_id={song})，预测分数={score:.4f}" for song, score in recommendations[:topk]]
    return result

# ItemCF
_sim_options_item = {'name': 'cosine', 'user_based': False}
_algo_itemcf = KNNBasic(sim_options=_sim_options_item)
_algo_itemcf.fit(_trainset)

def itemcf_topn(topk=10):
    # ItemCF推荐TopN
    user_id = str(_df_cf['user'].iloc[0])
    all_songs = _df_cf['song'].unique()
    user_listened = _df_cf[_df_cf['user'] == int(user_id)]['song'].tolist()
    to_predict = [song for song in all_songs if song not in user_listened]
    recommendations = []
    for song in to_predict[:100]:
        pred = _algo_itemcf.predict(user_id, song)
        recommendations.append((song, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    result = [f"{_song_info_dict.get(song, '未知')} (song_id={song})，预测分数={score:.4f}" for song, score in recommendations[:topk]]
    return result

# SVD
_algo_svd = SurpriseSVD()
_algo_svd.fit(_trainset)

def svd_topn(topk=10):
    # SVD推荐TopN
    user_id = str(_df_cf['user'].iloc[0])
    all_songs = _df_cf['song'].unique()
    user_listened = _df_cf[_df_cf['user'] == int(user_id)]['song'].tolist()
    to_predict = [song for song in all_songs if song not in user_listened]
    recommendations = []
    for song in to_predict[:100]:
        pred = _algo_svd.predict(user_id, song)
        recommendations.append((song, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    result = [f"{_song_info_dict.get(song, '未知')} (song_id={song})，预测分数={score:.4f}" for song, score in recommendations[:topk]]
    return result

# 深度学习推荐
_ncf_df = pd.read_csv(DATA_FILE, nrows=NCF_SAMPLE_SIZE)
_num_users = _ncf_df['user'].max() + 1
_num_songs = _ncf_df['song'].max() + 1
_ncf_model = NCF(_num_users, _num_songs)
try:
    _ncf_model.load_state_dict(torch.load(NCF_MODEL_PATH, map_location='cpu'))
    _ncf_model.eval()
except FileNotFoundError:
    print(f"警告: 模型文件 {NCF_MODEL_PATH} 不存在，深度学习推荐功能将不可用")
    _ncf_model = None

_ncf_song_info = pd.read_csv(DATA_FILE, usecols=['song', 'artist_name', 'title']).drop_duplicates('song').set_index('song')

def train_personalized_ncf(user_id, user_history, topk=10, n_epochs=3):
    # 基于用户历史记录训练个性化NCF模型
    if not user_history:
        return []
    
    try:
        from src.deep_learning_recommend import NCF, MusicDataset
        from torch.utils.data import DataLoader
        
        # 准备训练数据
        df_global = pd.read_csv(DATA_FILE, nrows=NCF_SAMPLE_SIZE)
        df_global = df_global[['user', 'song', 'play_count']]
        
        # 创建用户历史数据
        user_data = pd.DataFrame({
            'user': [user_id] * len(user_history),
            'song': user_history,
            'play_count': [1.0] * len(user_history)
        })
        
        # 合并用户数据和全局数据
        df_train = pd.concat([user_data, df_global.head(1000)], ignore_index=True)
        
        # 创建模型
        num_users = max(df_train['user'].max() + 1, user_id + 1)
        num_songs = df_train['song'].max() + 1
        
        model = NCF(num_users, num_songs, emb_dim=16)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # 训练模型
        train_dataset = MusicDataset(df_train)
        train_loader = DataLoader(train_dataset, batch_size=min(64, len(df_train)), shuffle=True)
        
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for user, song, label in train_loader:
                user, song, label = user.to(device), song.to(device), label.to(device)
                pred = model(user, song)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # 生成推荐
        model.eval()
        all_songs = df_global['song'].unique()
        user_listened = set(user_history)
        candidate_songs = [s for s in all_songs if s not in user_listened]
        
        if len(candidate_songs) == 0:
            return []
        
        # 预测候选歌曲分数
        user_tensor = torch.tensor([user_id] * len(candidate_songs), dtype=torch.long).to(device)
        song_tensor = torch.tensor(candidate_songs, dtype=torch.long).to(device)
        
        with torch.no_grad():
            scores = model(user_tensor, song_tensor).cpu().numpy()
        
        # 排序并取TopK
        song_score_pairs = list(zip(candidate_songs, scores))
        song_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_recommend = song_score_pairs[:topk]
        
        # 格式化结果
        result = []
        for sid, score in top_recommend:
            if sid in _ncf_song_info.index:
                artist = _ncf_song_info.loc[sid, 'artist_name']
                title = _ncf_song_info.loc[sid, 'title']
                result.append(f"{artist} - {title} (song_id={sid})，预测分数={score:.4f}")
            else:
                result.append(f"song_id={sid}，预测分数={score:.4f}")
        
        return result
        
    except Exception as e:
        # 训练失败，返回空列表
        import traceback
        print(f"个性化训练失败: {str(e)}")
        print(traceback.format_exc())
        return []


def ncf_recommend(user_id, topk=10, user_history=None):
    # NCF深度学习推荐TopN
    max_user_id = _num_users - 1
    
    # 如果用户ID超出范围且有历史记录，使用个性化训练
    if user_id > max_user_id and user_history:
        return train_personalized_ncf(user_id, user_history, topk)
    
    # 使用预训练模型
    if _ncf_model is None:
        raise FileNotFoundError(f"模型文件 {NCF_MODEL_PATH} 不存在，请先训练模型")
    
    user_tensor = torch.tensor([user_id]*_num_songs, dtype=torch.long)
    song_tensor = torch.arange(_num_songs, dtype=torch.long)
    with torch.no_grad():
        scores = _ncf_model(user_tensor, song_tensor).numpy()
    top_idx = scores.argsort()[::-1][:topk]
    result = []
    for sid in top_idx:
        if sid in _ncf_song_info.index:
            artist = _ncf_song_info.loc[sid, 'artist_name']
            title = _ncf_song_info.loc[sid, 'title']
            result.append(f"{artist} - {title} (song_id={sid})，预测分数={scores[sid]:.4f}")
        else:
            result.append(f"song_id={sid}，预测分数={scores[sid]:.4f}")
    return result 

@st.cache_data
def _get_content_features():
    # 缓存内容特征
    df = pd.read_csv(DATA_FILE)
    content_features = ['artist_name', 'title', 'year', 'artist_familiarity', 'artist_hotttnesss']
    songs = df.drop_duplicates('song')[['song'] + content_features].set_index('song')
    return songs

@st.cache_data
def _get_feature_encoder():
    # 缓存特征编码器
    songs = _get_content_features()
    # 创建特征编码器
    artist_hasher = FeatureHasher(n_features=50, input_type='string')
    scaler = MinMaxScaler()
    
    # 准备数据
    artist_names = songs['artist_name'].fillna('unknown').astype(str).values
    num_features = songs[['year', 'artist_familiarity', 'artist_hotttnesss']].fillna(0).values
    
    # 训练编码器
    artist_list = [[name] for name in artist_names]
    artist_features = artist_hasher.transform(artist_list).toarray()
    num_features_scaled = scaler.fit_transform(num_features)
    
    return artist_hasher, scaler, songs

def _get_song_feature(song_id, artist_hasher, scaler, songs):
    # 获取单个歌曲的特征向量
    if song_id not in songs.index:
        return None
    
    row = songs.loc[song_id]
    artist_name = str(row['artist_name']) if pd.notna(row['artist_name']) else 'unknown'
    artist_feature = artist_hasher.transform([[artist_name]]).toarray()[0]
    
    num_values = [[row['year'] if pd.notna(row['year']) else 0,
                   row['artist_familiarity'] if pd.notna(row['artist_familiarity']) else 0,
                   row['artist_hotttnesss'] if pd.notna(row['artist_hotttnesss']) else 0]]
    num_feature = scaler.transform(num_values)[0]
    
    return np.hstack([artist_feature, num_feature])

def content_based_recommend(user_history_songs, topk=10):
    # 基于用户历史的内容推荐
    if not user_history_songs:
        return []
    
    # 获取特征编码器和歌曲数据
    artist_hasher, scaler, songs = _get_feature_encoder()
    
    # 计算用户画像
    user_profile_features = []
    for song_id in user_history_songs:
        feature = _get_song_feature(song_id, artist_hasher, scaler, songs)
        if feature is not None:
            user_profile_features.append(feature)
    
    if not user_profile_features:
        return []
    
    user_profile = np.mean(user_profile_features, axis=0)
    
    # 计算候选歌曲与用户画像的相似度
    user_listened = set(user_history_songs)
    candidates = []
    
    candidate_songs = [sid for sid in songs.index if sid not in user_listened]
    # 如果候选歌曲太多，随机采样
    if len(candidate_songs) > 10000:
        import random
        candidate_songs = random.sample(candidate_songs, 10000)
    
    for song_id in candidate_songs:
        song_feature = _get_song_feature(song_id, artist_hasher, scaler, songs)
        if song_feature is not None:
            # 计算相似度
            similarity = np.dot(user_profile, song_feature) / (
                np.linalg.norm(user_profile) * np.linalg.norm(song_feature) + 1e-8
            )
            candidates.append((song_id, similarity))
    
    # 排序并取TopK
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_recommend = candidates[:topk]
    
    result = []
    for song_id, score in top_recommend:
        row = songs.loc[song_id]
        result.append(f"{row['artist_name']} - {row['title']} (song_id={song_id})，相似度：{score:.4f}")
    return result


def extract_song_id_from_result(result_str):
    # 从推荐结果字符串中提取song_id
    import re
    match = re.search(r'song_id=(\d+)', result_str)
    if match:
        return int(match.group(1))
    return None


def get_max_user_id_for_hybrid():
    # 获取数据中的最大用户ID
    try:
        df = pd.read_csv(DATA_FILE, usecols=['user'], nrows=NCF_SAMPLE_SIZE)
        return int(df['user'].max())
    except:
        return 1000


def hybrid_recommend(user_id, user_history, topk=10, weights=None):
    # 融合推荐算法（加权融合）
    if weights is None:
        # 根据用户类型自动选择权重
        history_count = len(user_history) if user_history else 0
        max_user_id = get_max_user_id_for_hybrid()
        
        if history_count == 0:
            # 新用户：主要使用冷启动和CF
            weights = {'cf': 0.5, 'ncf': 0.3, 'content': 0.2}
        elif history_count < 5:
            # 少量历史：CF和内容推荐为主
            if user_id is not None and user_id <= max_user_id:
                weights = {'cf': 0.4, 'ncf': 0.3, 'content': 0.3}
            else:
                weights = {'cf': 0.5, 'ncf': 0.0, 'content': 0.5}
        else:
            # 活跃用户：均衡使用所有算法
            if user_id is not None and user_id <= max_user_id:
                weights = {'cf': 0.4, 'ncf': 0.4, 'content': 0.2}
            else:
                weights = {'cf': 0.5, 'ncf': 0.0, 'content': 0.5}
    
    # 获取各算法的推荐结果
    song_scores = {}
    
    # 协同过滤推荐
    try:
        cf_results = svd_topn(topk * 2)
        for result_str in cf_results:
            song_id = extract_song_id_from_result(result_str)
            if song_id:
                # 提取分数
                import re
                score_match = re.search(r'预测分数=([\d.]+)', result_str)
                if score_match:
                    score = float(score_match.group(1))
                    if song_id not in song_scores:
                        song_scores[song_id] = {}
                    song_scores[song_id]['cf'] = score
                    song_scores[song_id]['cf_str'] = result_str
    except:
        pass
    
    # 深度学习推荐
    if weights.get('ncf', 0) > 0:
        try:
            ncf_results = ncf_recommend(user_id, topk * 2, user_history)
            for result_str in ncf_results:
                song_id = extract_song_id_from_result(result_str)
                if song_id:
                    import re
                    score_match = re.search(r'预测分数=([\d.]+)', result_str)
                    if score_match:
                        score = float(score_match.group(1))
                        if song_id not in song_scores:
                            song_scores[song_id] = {}
                        song_scores[song_id]['ncf'] = score
                        song_scores[song_id]['ncf_str'] = result_str
        except Exception as e:
            # NCF推荐失败，跳过
            pass
    
    # 内容推荐
    if weights.get('content', 0) > 0 and user_history:
        try:
            content_results = content_based_recommend(user_history, topk * 2)
            for result_str in content_results:
                song_id = extract_song_id_from_result(result_str)
                if song_id:
                    import re
                    score_match = re.search(r'相似度：([\d.]+)', result_str)
                    if score_match:
                        score = float(score_match.group(1))
                        if song_id not in song_scores:
                            song_scores[song_id] = {}
                        song_scores[song_id]['content'] = score
                        song_scores[song_id]['content_str'] = result_str
        except:
            pass
    
    # 归一化各算法的分数到[0, 1]范围
    for song_id in song_scores:
        scores = song_scores[song_id]
        if 'cf' in scores:
            scores['cf'] = min(1.0, max(0.0, scores['cf'] / 5.0))
        if 'ncf' in scores:
            scores['ncf'] = min(1.0, max(0.0, scores['ncf'] / 5.0))
    
    # 计算融合分数
    fusion_scores = []
    for song_id, scores in song_scores.items():
        fusion_score = 0.0
        if 'cf' in scores:
            fusion_score += weights.get('cf', 0) * scores['cf']
        if 'ncf' in scores:
            fusion_score += weights.get('ncf', 0) * scores['ncf']
        if 'content' in scores:
            fusion_score += weights.get('content', 0) * scores['content']
        
        # 获取歌曲信息
        song_info = scores.get('cf_str') or scores.get('ncf_str') or scores.get('content_str', f'song_id={song_id}')
        # 提取歌手和歌名
        import re
        info_match = re.search(r'(.+?)\s*\(song_id=', song_info)
        if info_match:
            song_name = info_match.group(1).strip()
        else:
            song_name = f'song_id={song_id}'
        
        fusion_scores.append((song_id, fusion_score, song_name))
    
    # 排序并取TopK
    fusion_scores.sort(key=lambda x: x[1], reverse=True)
    top_recommend = fusion_scores[:topk]
    
    # 格式化结果
    result = []
    for song_id, score, song_name in top_recommend:
        result.append(f"{song_name} (song_id={song_id})，融合分数={score:.4f}")
    
    return result


