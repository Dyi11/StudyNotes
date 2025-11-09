"""
一个简化版的 GPT 类模型
"""
import os          # 用于文件路径操作
import requests    # 用于下载数据
import math        # 数学运算（如sqrt、log）
import tiktoken    # OpenAI的分词器（与GPT-3相同）
import torch       # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from torch.nn import functional as F  # 神经网络函数（如softmax、交叉熵）

# 超参数
batch_size = 4          # 每个训练步骤的样本数（批次大小）
context_length = 16     # 每个样本的token长度（上下文窗口大小）
d_model = 64            # 模型中token嵌入的维度（核心维度，类似GPT的隐藏层大小）
num_blocks = 8          # Transformer块的数量（模型深度）
num_heads = 4           # 多头注意力中的头数
learning_rate = 1e-3    # 学习率（0.001）
dropout = 0.1           #  dropout概率（防止过拟合）
max_iters = 2000        # 训练总迭代次数（测试时可减小）
eval_interval = 50      # 每隔多少步评估一次损失
eval_iters = 20         # 评估时的平均迭代次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备（优先GPU）
TORCH_SEED = 1337       # 随机种子（保证结果可复现）
torch.manual_seed(TORCH_SEED)  # 设置随机种子

# 定义文件路径（使用绝对路径更稳妥，避免相对路径混淆）
file_path = os.path.join(os.path.dirname(__file__), "data", "sales_textbook.txt")
# 上面的代码会自动拼接当前脚本所在目录 + data文件夹 + 文件名
# 例如：如果脚本在C:\test\model.py，则file_path为C:\test\data\sales_textbook.txt

# 检查文件是否存在
if not os.path.exists(file_path):
    # 1. 先创建data文件夹（确保目录存在）
    data_dir = os.path.dirname(file_path)  # 提取data文件夹的绝对路径
    os.makedirs(data_dir, exist_ok=True)   # 自动创建，已存在则不报错
    
    # 2. 用绝对路径写入文件（和前面判断的路径完全一致）
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'  # 替换为实际URL
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # 这里必须用file_path，而不是相对路径
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"文件已保存至：{file_path}")
    except Exception as e:
        print(f"下载失败：{e}")

# 读取文本内容
with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 文本分词（Tokenization），分词是将文本转换为模型可处理的数字序列的过程
# 使用tiktoken的cl100k_base编码（与GPT-3相同的分词方式）
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)  # 将文本转换为token（整数列表）
max_token_value = max(tokenized_text) + 1  # 最大token值（用于嵌入层维度），token的总数
# 转换为PyTorch张量，并移动到指定设备（GPU/CPU）
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# 划分数据集
split_idx = int(len(tokenized_text) * 0.9)  # 90%作为训练集，10%作为验证集
train_data = tokenized_text[:split_idx]     # 训练数据（前90%）
val_data = tokenized_text[split_idx:]       # 验证数据（后10%）

# 定义 FeedForward 网络（前馈网络）
# 前馈网络是 Transformer 块的一部分，作用是对每个 token 的特征进行非线性变换（先升维再降维，增加模型表达能力）
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化父类
        self.d_model = d_model
        self.dropout = dropout
        # 定义前馈网络：线性层->ReLU激活->线性层->dropout，标准型FFN
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),  # 升维4倍
            nn.ReLU(),  # 非线性激活
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),  # 降维回d_model
            nn.Dropout(dropout),  # 随机失活（防止过拟合）
        )

    def forward(self, x):
        return self.ffn(x)  # 前向传播：直接调用定义的序列


# 定义 Scaled Dot Product Attention（缩放点积注意力）
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size  # 每个注意力头的维度
        self.context_length = context_length
        self.dropout = dropout

        # 定义Q、K、V的线性投影层（无偏置）
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        # 注册下三角掩码（用于因果注意力，防止模型看到未来的token）
        self.register_buffer('tril', torch.tril(torch.ones((self.context_length, self.context_length))))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # B=批次大小，T=时间步（上下文长度），C=d_model
        assert T <= self.context_length  # 确保输入长度不超过预设上下文长度
        assert C == self.d_model         # 确保输入维度正确

        # 计算Q、K、V（通过线性层投影）
        q = self.query_layer(x)  # (B, T, head_size)
        k = self.key_layer(x)    # (B, T, head_size)
        v = self.value_layer(x)  # (B, T, head_size)

        # 计算注意力权重：Q·K^T / sqrt(d_k)（缩放防止梯度消失）
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T)
        # 应用掩码：将上三角部分（未来token）设为负无穷（softmax后为0）
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 归一化权重（softmax确保和为1）
        weights = F.softmax(input=weights, dim=-1)
        # dropout进一步防止过拟合
        weights = self.dropout_layer(weights)

        # 注意力输出：权重·V（加权求和）
        out = weights @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 并行计算所有单头注意力的输出，并在最后一个维度拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size) = (B, T, d_model)
        # 2. 通过投影层融合多头信息
        out = self.projection_layer(out)  # (B, T, d_model)
        # 3. 应用dropout防止过拟合
        out = self.dropout_layer(out)     # (B, T, d_model)
        return out
    

class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # Note: The order of the operations is different from the original Transformer paper
        # The order here is: LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # Residual connection
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # 1. 词嵌入表：将token索引映射为d_model维度的向量
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # Run all the transformer blocks
        # Different from original paper, here we add a final layer norm after all the blocks
        # 2. Transformer块堆叠：包含num_blocks个TransformerBlock，最后加一层LayerNorm
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        # 3. 语言模型输出层：将d_model维度映射为词汇表大小（预测下一个token）
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape # B=批次大小，T=当前序列长度（<= context_length）
        
        # 1. 位置编码（遵循Transformer原始论文的正弦余弦编码）
        # 创建位置编码表（context_length, d_model）
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1) # (context_length, 1)

        # 计算角度的分母项（用于正弦余弦编码）
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # 偶数维度用正弦，奇数维度用余弦
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 0,2,4...维度
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 1,3,5...维度

        # 截取与当前序列长度T匹配的位置编码，并移动到设备（GPU/CPU）
        position_embedding = position_encoding_lookup_table[:T, :].to(device)

        # 2. 词嵌入 + 位置编码（逐元素相加）
        x = self.token_embedding_lookup_table(idx) + position_embedding

        # 3. 通过所有Transformer块
        x = self.transformer_blocks(x)  # (B, T, d_model)

        # 4. 输出层：计算logits（未归一化的预测得分）
        logits = self.language_model_out_linear_layer(x)  # (B, T, max_token_value)

        # 5. 计算损失（如果提供了目标targets）
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            
            idx_crop = idx[:, -self.context_length:] # 截取序列的最后 self.context_length 个 token
            # 获取模型预测
            # # logits形状：(B, T_crop, C)，C是词汇表大小
            logits, loss = self(idx_crop)
            # 只关心当前上下文最后一个位置对应的下一个 token 预测，因此提取logits的最后一个时间步（-1）
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # torch.multinomial根据概率分布随机采样 1 个 token（num_samples=1）
            # 实现 “随机生成”（而非总是选概率最高的 token，避免生成结果单调)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将新生成的idx_next拼接到原始序列idx的末尾（dim=1表示在序列长度维度拼接）
            # 更新上下文序列，为下一轮生成做准备
            idx = torch.cat((idx, idx_next), dim=1)
        return idx # 形状：(B, T + max_new_tokens)


# Initialize the model
model = TransformerLanguageModel()
model = model.to(device)


# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))

    # 将多个长度为 context_length 的一维序列堆叠成二维张量，形状为 (batch_size, context_length)
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    # 目标序列 y 是输入序列 x 向右偏移一位的结果
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
# 在不更新模型参数的情况下，估算模型在训练集和验证集上的平均损失
def estimate_loss():
    out = {}
    model.eval()
    # 在训练过程中定期调用该函数，计算模型在训练集（train）和验证集（valid）上的平均损失
    for split in ['train', 'valid']:
        # eval_iters 的作用：指定计算损失时的迭代次数（例如 5 次）
        losses = torch.zeros(eval_iters)  # 存储多次迭代的损失
        for k in range(eval_iters):
            # 获取一批数据（输入x和目标y）
            x_batch, y_batch = get_batch(split)
            # 前向传播计算logits和损失（不计算梯度）
            logits, loss = model(x_batch, y_batch)
            # 记录当前批次的损失值（转为Python数值）
            losses[k] = loss.item()
            # 计算该数据集的平均损失并存储
        out[split] = losses.mean()
    model.train()
    return out


# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    # 每次迭代通过 “前向传播→梯度清零→反向传播→参数更新” 完成一次参数优化
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
# 使用训练好的 Transformer 语言模型进行文本生成的功能，
# 以指定字符串'The salesperson'为起始文本，自动生成后续内容
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
