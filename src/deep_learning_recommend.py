# 深度学习推荐模块
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_FILE, NCF_MODEL_PATH, NCF_SAMPLE_SIZE


# NCF模型
class NCF(nn.Module):
    # 神经协同过滤模型
    def __init__(self, num_users, num_songs, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.song_emb = nn.Embedding(num_songs, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, user, song):
        u = self.user_emb(user)
        s = self.song_emb(song)
        x = torch.cat([u, s], dim=1)
        out = self.mlp(x)
        return out.squeeze()


# 数据集类
class MusicDataset(Dataset):
    # 音乐推荐数据集
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.songs = torch.tensor(df['song'].values, dtype=torch.long)
        self.labels = torch.tensor(df['play_count'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.labels[idx]


def train_ncf_model(n_epochs=5, batch_size=1024, lr=0.001):
    # 训练NCF模型
    # 读取数据
    print(f'正在读取数据（前{NCF_SAMPLE_SIZE}条）...')
    df = pd.read_csv(DATA_FILE, nrows=NCF_SAMPLE_SIZE)
    df = df[['user', 'song', 'play_count']]
    
    # 划分训练集和测试集
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # 构建Dataset
    train_dataset = MusicDataset(train)
    test_dataset = MusicDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    num_users = df['user'].max() + 1
    num_songs = df['song'].max() + 1
    model = NCF(num_users, num_songs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 训练模型
    print('开始训练模型...')
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for user, song, label in train_loader:
            user, song, label = user.to(device), song.to(device), label.to(device)
            pred = model(user, song)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(user)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {total_loss/len(train_dataset):.4f}")

    # 测试RMSE
    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for user, song, label in test_loader:
            user, song = user.to(device), song.to(device)
            pred = model(user, song).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
        rmse = ((np.array(preds) - np.array(labels)) ** 2).mean() ** 0.5
        print(f"Test RMSE: {rmse:.4f}")

    # 保存模型
    os.makedirs(os.path.dirname(NCF_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), NCF_MODEL_PATH)
    print(f"模型已保存为 {NCF_MODEL_PATH}")
    
    return model


if __name__ == '__main__':
    # 直接运行此文件则训练模型
    train_ncf_model()
