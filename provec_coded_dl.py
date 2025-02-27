import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# Load the genotype_fitness_data.tsv file
file_path = 'C:/Users/Thomascrx/Desktop/ml_code/sequence_points_file.csv'  # Replace with your file path
genotype_fitness_data = pd.read_csv(file_path)
print(genotype_fitness_data.head())



# In[17]:


def split_overlapping(s, window_size):
    # 使用列表推导式生成重叠的子字符串
    return [s[i:i + window_size] for i in range(len(s) - window_size + 1)]

def cut_to_provec(sequence,num):
   return [split_overlapping(i, window_size=num) for i in sequence]
    
sequence=genotype_fitness_data["sequence"]
sequence=cut_to_provec(sequence,3)                              
provec=pd.read_csv("C:/Users/Thomascrx/Desktop/protVec_100d_3grams.csv",sep="\t")
# 指定要合并的列
columns=list(provec.columns)
columns.pop(0)
columns_to_merge = columns
# 对每一行操作，将指定列的值合并成一个列表，并存储到新列中
provec['Merged'] = provec[columns_to_merge].apply(lambda row: row.tolist(), axis=1)
# 删除已合并的列
provec.drop(columns=columns_to_merge, inplace=True)
provec_dict = provec.set_index('words')['Merged'].to_dict()

def provec_encode_aa(sequence):
    return np.array([provec_dict[aa] for aa in sequence])


X = np.array([provec_encode_aa(aa) for aa in sequence])
y = genotype_fitness_data['delta_log10Ka'].values
print(f"X shape: {X.shape}")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
print(f"X_train_tensor shape: {X_train_tensor.shape}")
# 标准化 y（使用训练集的均值和标准差）
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
print(f"y_train_tensor shape after view: {y_train_tensor.shape}")

# Create DataLoader for training and testing
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
# 配置参数
class Config:
    seq_len = 201          # 序列长度
    vocab_size = 20       # 氨基酸种类数（独热编码维度）
    d_model = 256         # 嵌入维度
    nhead = 8             # 注意力头数
    num_layers = 4        # 编码器层数
    dim_feedforward = 1024 # 前馈网络维度
    dropout = 0.1         # Dropout 比率
    batch_size = 64       # Batch Size（根据 GPU 显存调整）
    lr = 3e-4             # 学习率
    weight_decay = 0.01   # 权重衰减
    epochs = 100          # 训练轮数
    grad_clip = 5.0       # 梯度裁剪阈值
    log_dir = "runs/exp1" # TensorBoard 日志目录

# 模型定义
class ProteinTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(config.vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.seq_len * config.d_model, 1)

    def forward(self, x):
        x = self.embedding(x)          # (batch, 201, 20) → (batch, 201, 256)
        x = x.permute(1, 0, 2)         # (201, batch, 256)
        x = self.encoder(x)            # (201, batch, 256)
        x = x.permute(1, 0, 2)         # (batch, 201, 256)
        x = x.reshape(x.size(0), -1)   # (batch, 201 * 256=51456)
        return self.fc(x)              # (batch, 1)

# 初始化模型和优化器
config = Config()
model = ProteinTransformer(config)
model = model.to("cuda")

# 多 GPU 支持
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.MSELoss()
scaler = GradScaler()

# 初始化 TensorBoard 写入器
writer = SummaryWriter(log_dir=config.log_dir)

# 示例数据加载（假设 X, y 已预处理为 Tensor）
dataset = TensorDataset(torch.randn(136204, 201, 20), torch.randn(136204, 1))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

# 训练循环
for epoch in range(config.epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda")
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
        
        # 反向传播与梯度裁剪
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # 记录 batch 级损失
        current_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar("Loss/train (batch)", loss.item(), current_step)
        
        total_loss += loss.item()
    
    # 记录 epoch 级平均损失
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train (epoch)", avg_loss, epoch)
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# 关闭 TensorBoard 写入器
writer.close()