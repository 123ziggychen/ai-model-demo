import numpy as np
from collections import Counter, defaultdict
import re
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

class AIMetaModel:
    def __init__(self, vector_size=100, window_size=2, min_count=2):
        """
        初始化AI元模型
        
        参数:
            vector_size: 词向量维度
            window_size: 上下文窗口大小
            min_count: 词出现的最小次数
        """
        print(f"初始化模型参数: vector_size={vector_size}, window_size={window_size}, min_count={min_count}")
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        
        # 词汇表
        self.vocab = {}
        self.reverse_vocab = {}
        self.token_counts = Counter()
        
        # 词向量
        self.word_vectors = {}
        
        # 转移概率矩阵 P(next_token|current_token)
        self.transition_probs = defaultdict(Counter)
        
    def tokenize(self, text):
        """将文本分词为tokens"""
        # 简单的分词方法，使用空格、标点等作为分隔符
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def build_vocab(self, texts):
        """构建词汇表"""
        print("开始构建词汇表...")
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
            self.token_counts.update(tokens)
        
        # 过滤低频词
        valid_tokens = [token for token, count in self.token_counts.items() 
                       if count >= self.min_count]
        
        # 构建词汇表映射
        self.vocab = {token: idx for idx, token in enumerate(valid_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        print(f"词汇表构建完成，大小: {len(self.vocab)}")
        return self.vocab
    
    def vectorize(self, token):
        """将token向量化"""
        if token in self.word_vectors:
            return self.word_vectors[token]
        elif token in self.vocab:
            # 如果词在词汇表中但没有向量，生成随机向量
            vector = np.random.normal(0, 0.1, self.vector_size)
            self.word_vectors[token] = vector
            return vector
        else:
            # 未知词
            return np.zeros(self.vector_size)
    
    def compute_transition_probs(self, texts):
        """计算词间转移概率"""
        print("开始计算转移概率...")
        for text in texts:
            tokens = self.tokenize(text)
            for i in range(len(tokens) - 1):
                current = tokens[i]
                next_token = tokens[i + 1]
                if current in self.vocab and next_token in self.vocab:
                    self.transition_probs[current][next_token] += 1
        
        # 将计数转换为概率
        for token, next_tokens in self.transition_probs.items():
            total = sum(next_tokens.values())
            for next_token in next_tokens:
                self.transition_probs[token][next_token] /= total
        
        print("转移概率计算完成")
    
    def train(self, texts):
        """训练模型"""
        try:
            print("开始训练过程...")
            self.build_vocab(texts)
            
            print("初始化词向量...")
            # 初始为随机向量
            for token in self.vocab:
                self.word_vectors[token] = np.random.normal(0, 0.1, self.vector_size)
            
            self.compute_transition_probs(texts)
            
            print("训练完成!")
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            raise
    
    def predict_next_token(self, current_token, n=1):
        """预测下一个token"""
        if current_token not in self.transition_probs:
            # 如果当前token没有在训练集中出现过，返回随机选择
            possible_tokens = list(self.vocab.keys())
            next_tokens = np.random.choice(possible_tokens, n)
            return list(next_tokens)
        
        # 基于转移概率选择
        next_tokens_probs = self.transition_probs[current_token]
        tokens = []
        probs = []
        
        for token, prob in next_tokens_probs.items():
            tokens.append(token)
            probs.append(prob)
        
        if not tokens:  # 如果没有任何概率记录
            return [np.random.choice(list(self.vocab.keys())) for _ in range(n)]
        
        # 基于概率选择n个token
        next_tokens = np.random.choice(tokens, size=min(n, len(tokens)), 
                                      p=np.array(probs)/sum(probs), 
                                      replace=False)
        return list(next_tokens)
    
    def generate_text(self, start_token, length=10):
        """生成文本"""
        if start_token not in self.vocab:
            start_token = np.random.choice(list(self.vocab.keys()))
        
        generated = [start_token]
        current_token = start_token
        
        for _ in range(length - 1):
            next_token = self.predict_next_token(current_token, 1)[0]
            generated.append(next_token)
            current_token = next_token
        
        return " ".join(generated)
    
    def visualize_topology(self, n_components=3):
        """可视化词向量的拓扑结构"""
        print("开始生成拓扑可视化...")
        if not self.word_vectors:
            print("还没有训练词向量，无法可视化")
            return
        
        try:
            # 提取所有词向量
            tokens = list(self.word_vectors.keys())
            vectors = np.array([self.word_vectors[token] for token in tokens])
            
            # 使用PCA降维到3D
            pca = PCA(n_components=n_components)
            reduced_vectors = pca.fit_transform(vectors)
            
            # 3D可视化
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            xs = reduced_vectors[:, 0]
            ys = reduced_vectors[:, 1]
            zs = reduced_vectors[:, 2]
            
            ax.scatter(xs, ys, zs, alpha=0.5)
            
            # 标注部分词
            for i, token in enumerate(tokens[:20]):  # 只标注前20个词，避免过于拥挤
                ax.text(xs[i], ys[i], zs[i], token)
            
            ax.set_title("词向量拓扑结构 (PCA 3D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            
            plt.tight_layout()
            plt.savefig('word_topology.png')
            plt.show()
            print("拓扑可视化完成，已保存为word_topology.png")
        except Exception as e:
            print(f"可视化过程中发生错误: {str(e)}")
            raise
    
    def save(self, filename):
        """保存模型"""
        try:
            print(f"开始保存模型到 {filename}...")
            with open(filename, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'reverse_vocab': self.reverse_vocab,
                    'token_counts': self.token_counts,
                    'word_vectors': self.word_vectors,
                    'transition_probs': dict(self.transition_probs),
                    'vector_size': self.vector_size,
                    'window_size': self.window_size,
                    'min_count': self.min_count
                }, f)
            print(f"模型已成功保存到 {filename}")
        except Exception as e:
            print(f"保存模型时发生错误: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filename):
        """加载模型"""
        try:
            print(f"开始从 {filename} 加载模型...")
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            model = cls(
                vector_size=data['vector_size'],
                window_size=data['window_size'],
                min_count=data['min_count']
            )
            
            model.vocab = data['vocab']
            model.reverse_vocab = data['reverse_vocab']
            model.token_counts = data['token_counts']
            model.word_vectors = data['word_vectors']
            
            # 转换回defaultdict(Counter)
            model.transition_probs = defaultdict(Counter)
            for token, counter in data['transition_probs'].items():
                model.transition_probs[token].update(counter)
            
            print(f"模型已成功从 {filename} 加载")
            return model
        except Exception as e:
            print(f"加载模型时发生错误: {str(e)}")
            raise 