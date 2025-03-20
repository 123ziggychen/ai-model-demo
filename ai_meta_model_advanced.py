import numpy as np
from collections import Counter, defaultdict
import re
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import jieba
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerCN:
    """中文分词器"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.word_freq = Counter()
        self.fitted = False
    
    def fit(self, texts):
        """训练分词器"""
        logger.info("开始训练分词器...")
        for text in texts:
            words = list(jieba.cut(text))
            self.word_freq.update(words)
        
        # 选择最常见的词构建词汇表
        common_words = [word for word, _ in self.word_freq.most_common(self.vocab_size - 4)]
        for i, word in enumerate(common_words):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word
        
        self.fitted = True
        logger.info(f"分词器训练完成，词汇表大小: {len(self.word2idx)}")
    
    def tokenize(self, text):
        """将文本分词为token列表"""
        if not self.fitted:
            raise ValueError("分词器尚未训练")
        
        words = list(jieba.cut(text))
        return words
    
    def encode(self, text, add_special_tokens=True):
        """将文本编码为索引列表"""
        tokens = self.tokenize(text)
        
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx["<BOS>"])
            
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])
                
        if add_special_tokens:
            indices.append(self.word2idx["<EOS>"])
            
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """将索引列表解码为文本"""
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] if skip_special_tokens else []
        
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if token not in special_tokens:
                    tokens.append(token)
            else:
                tokens.append("<UNK>")
                
        return "".join(tokens)
    
    def save(self, path):
        """保存分词器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'vocab_size': self.vocab_size,
                'fitted': self.fitted
            }, f)
            
    @classmethod
    def load(cls, path):
        """加载分词器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer.word_freq = data['word_freq']
        tokenizer.fitted = data['fitted']
        
        return tokenizer

class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        
        for text in texts:
            encoded = self.tokenizer.encode(text)
            if len(encoded) < 3:  # 至少需要BOS、一个词和EOS
                continue
                
            # 截断或填充序列
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded = encoded + [0] * (max_length - len(encoded))
                
            self.inputs.append(encoded[:-1])  # 输入不包含EOS
            self.targets.append(encoded[1:])  # 目标不包含BOS
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # 自注意力
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        out1 = self.norm1(x + attention_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AIMetaModelAdvanced(nn.Module):
    """改进版AI元模型，使用Transformer架构"""
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=512, 
                 num_layers=4, max_length=100, dropout=0.1):
        super(AIMetaModelAdvanced, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # 输入嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, seq_len):
        """生成掩码，用于自回归生成"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # 构建掩码
        mask = self.generate_mask(seq_len).to(x.device)
        
        # 嵌入和位置编码
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)
            
        # 输出层
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        output = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def predict_next_token(self, context, k=3):
        """预测下一个token"""
        with torch.no_grad():
            self.eval()
            
            # 将上下文转换为模型输入
            context_tensor = torch.tensor(context).unsqueeze(0)  # [1, seq_len]
            
            # 前向传播
            logits = self.forward(context_tensor)  # [1, seq_len, vocab_size]
            
            # 获取最后一个token的预测
            last_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # 应用softmax获取概率
            probabilities = torch.softmax(last_token_logits, dim=0)
            
            # 选择概率最高的k个token
            topk_probs, topk_indices = torch.topk(probabilities, k)
            
            return topk_indices.tolist(), topk_probs.tolist()
    
    def generate_text(self, tokenizer, start_text, max_length=50, temperature=1.0, top_k=0, top_p=0.0):
        """生成文本
        
        Args:
            tokenizer: 分词器
            start_text: 起始文本或token列表
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: 仅考虑概率最高的k个token，0表示禁用
            top_p: 仅考虑累积概率超过p的token，0表示禁用
        """
        # 将输入文本转换为token索引
        if isinstance(start_text, str):
            context = tokenizer.encode(start_text)
        else:
            context = start_text
            
        # 确保不超过最大长度
        if len(context) > self.max_length:
            context = context[:self.max_length]
            
        # 生成文本
        generated = context
        
        for _ in range(max_length):
            # 预测下一个token
            next_token_indices, next_token_probs = self.predict_next_token(generated, k=max(top_k, 10) if top_k > 0 else 100)
            
            # 转换为tensor
            indices = torch.tensor(next_token_indices)
            probs = torch.tensor(next_token_probs)
            
            # 应用温度
            if temperature != 1.0:
                probs = torch.pow(probs, 1.0 / temperature)
                probs = probs / probs.sum()
            
            # 应用top_p（核采样）
            if top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                
                # 移除累积概率超过p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0] = False  # 保留概率最高的token
                
                # 创建掩码
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                
                # 重新归一化概率
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # 应用top_k
            if top_k > 0:
                # 保留概率最高的k个token
                topk_probs, topk_indices = torch.topk(probs, min(top_k, len(probs)))
                
                # 创建新的概率分布
                probs = torch.zeros_like(probs)
                probs.scatter_(0, topk_indices, topk_probs)
                
                # 重新归一化概率
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # 根据概率选择下一个token
            try:
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token = next_token_indices[next_token_idx]
            except:
                # 如果采样失败，选择概率最高的token
                next_token = next_token_indices[0]
            
            # 添加到生成的序列
            generated.append(next_token)
            
            # 如果生成了EOS，停止生成
            if next_token == tokenizer.word2idx["<EOS>"]:
                break
            
            # 如果序列太长，截断
            if len(generated) >= self.max_length:
                break
        
        # 解码生成的序列
        return tokenizer.decode(generated)

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"使用设备: {self.device}")
    
    def train(self, texts, batch_size=32, epochs=5, lr=3e-4, save_path=None):
        """训练模型"""
        logger.info("准备训练数据...")
        dataset = TextDataset(texts, self.tokenizer, max_length=self.model.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD的损失
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        # 训练循环
        best_loss = float('inf')
        logger.info(f"开始训练，共 {epochs} 个epoch...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, targets in progress_bar:
                # 将数据移动到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = criterion(
                    outputs.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1)
                )
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # 更新进度条
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss/(progress_bar.n+1))
            
            # 每个epoch结束后计算平均损失
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            # 调整学习率
            scheduler.step(avg_loss)
            
            # 保存最佳模型
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_path)
                logger.info(f"保存最佳模型到 {save_path}")
    
    def evaluate(self, texts, batch_size=32):
        """评估模型"""
        self.model.eval()
        
        dataset = TextDataset(texts, self.tokenizer, max_length=self.model.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(
                    outputs.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def visualize_embeddings(self, n_components=3):
        """可视化词嵌入的拓扑结构"""
        # 提取嵌入矩阵
        embeddings = self.model.embedding.weight.detach().cpu().numpy()
        
        # 使用PCA降维
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 只可视化前200个词，避免过于拥挤
        n_words = min(200, len(self.tokenizer.idx2word))
        
        xs = reduced_embeddings[:n_words, 0]
        ys = reduced_embeddings[:n_words, 1]
        zs = reduced_embeddings[:n_words, 2]
        
        ax.scatter(xs, ys, zs, alpha=0.5)
        
        # 添加词标签
        for i in range(n_words):
            if i in self.tokenizer.idx2word:
                word = self.tokenizer.idx2word[i]
                ax.text(xs[i], ys[i], zs[i], word)
        
        ax.set_title("词嵌入拓扑结构 (PCA 3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        
        plt.tight_layout()
        plt.savefig('word_embeddings_topology.png')
        plt.show()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
            'max_length': self.model.max_length
        }, path)
    
    @classmethod
    def load_model(cls, path, tokenizer):
        """加载模型"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        model = AIMetaModelAdvanced(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            max_length=checkpoint['max_length']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer) 