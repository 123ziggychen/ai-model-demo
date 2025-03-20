# AI智障模型系统使用指南

## 项目介绍

AI智障模型系统是一个基于Transformer架构的中文自然语言处理的空白项目，包含模型训练、数据收集和交互界面三大核心功能。该系统能够通过学习文本数据，生成符合语境的文本内容，实现人机对话功能。（但这还只是一个demo，训练参数基本可以忽略不计，需要你自己寻找素材并且训练，因此如果你直接运行这个模型，那就只会阿巴阿巴）

## 系统模块说明

### 1. 核心模型 (`ai_meta_model_advanced.py`)

这是整个系统的核心文件，实现了AI模型的基础架构：

- **TokenizerCN**：中文分词器，负责将文本分解为词汇单元，构建词汇表
- **TextDataset**：文本数据集类，处理和准备训练数据
- **TransformerBlock**：实现Transformer的编码器块，包含自注意力机制和前馈网络
- **PositionalEncoding**：位置编码，为文本序列中的每个位置提供位置信息
- **AIMetaModelAdvanced**：完整的AI模型，整合上述组件，提供文本生成功能
- **ModelTrainer**：训练工具类，负责模型训练、评估和可视化

### 2. 训练模块 (`advanced_trainer.py`)

提供了模型训练的完整流程：

- 加载现有训练数据或通过爬虫收集新数据
- 创建或加载分词器
- 构建模型并进行训练
- 评估模型性能
- 可视化词嵌入（展示词汇之间的语义关系）
- 生成文本样本

### 3. 聊天界面 (`advanced_chat.py`)

用户友好的交互界面：

- 加载训练好的模型和分词器
- 提供命令行聊天界面
- 支持参数调整（温度、生成长度等）
- 提供多种命令（查看帮助、显示参数等）

### 4. 网络爬虫 (`web_crawler.py`)

负责从网络收集训练数据：

- 从指定网站抓取文本内容
- 清洗和处理文本数据
- 保存为训练文件

## 安装指南

### 前提条件

- Python 3.6或更高版本
- 安装pip包管理器

### 安装步骤

1. 下载或克隆项目代码
2. 创建并激活虚拟环境（推荐）：
   ```bash
   python -m venv ai_env
   # Windows
   ai_env\Scripts\activate
   # MacOS/Linux
   source ai_env/bin/activate
   ```
3. 安装依赖库：
   ```bash
   pip install torch numpy matplotlib scikit-learn tqdm jieba
   ```

## 使用指南

### 1. 准备训练数据

你可以选择使用现有的训练数据文件，或者通过爬虫收集新数据：

```bash
python advanced_trainer.py --url "https://example.com" --max_pages 50
```

### 2. 训练模型

使用以下命令训练模型：

```bash
python advanced_trainer.py --epochs 100 --vocab_size 10000 --visualize
```

参数说明：
- `--epochs`：训练轮数，值越大训练效果越好，但耗时越长
- `--vocab_size`：词汇表大小
- `--visualize`：启用此选项可以生成词嵌入可视化图
- `--generate`：训练后生成文本样本
- `--continue_training`：继续训练现有模型

### 3. 启动聊天界面

训练完成后，使用以下命令启动聊天界面：

```bash
python advanced_chat.py
```

在聊天界面中，你可以：
- 输入文本与AI模型对话
- 使用`/help`查看帮助
- 使用`/params`查看当前生成参数
- 使用`/set`调整参数，例如：`/set temperature 0.8`
- 使用`/quit`退出程序

参数调整说明：
- `temperature`：控制生成文本的创造性，值越高结果越随机多样
- `max_length`：生成文本的最大长度
- `top_k`：限制每次只考虑概率最高的k个词
- `top_p`：使用核采样方法控制生成多样性

## 故障排除

### 常见问题

1. **ModuleNotFoundError**：确保已安装所有依赖库
   ```bash
   pip install torch numpy matplotlib scikit-learn tqdm jieba
   ```

2. **内存不足**：如果训练数据过大，尝试减小`batch_size`
   ```bash
   python advanced_trainer.py --batch_size 16
   ```

3. **生成结果质量差**：
   - 增加训练轮数：`--epochs 200`
   - 增加训练数据量
   - 调整生成参数：在聊天界面中使用`/set temperature 0.7`等命令

4. **训练过慢**：
   - 减小模型规模：编辑`train_model`函数中的模型参数
   - 使用较小的训练数据集

## 高级功能

### 自定义模型结构

你可以修改`advanced_trainer.py`中的模型创建部分来调整模型结构：

```python
model = AIMetaModelAdvanced(
    vocab_size=len(tokenizer.word2idx),
    embed_dim=256,      # 嵌入维度
    num_heads=8,        # 注意力头数
    ff_dim=1024,        # 前馈网络维度
    num_layers=8,       # Transformer层数
    max_length=100,     # 最大序列长度
    dropout=0.2         # Dropout率
)
```

### 可视化分析

训练时添加`--visualize`参数，可以生成词嵌入的3D可视化图，帮助理解模型学到的语义关系。

## 性能提示

1. 如果有GPU，系统将自动使用它加速训练
2. 增加训练数据量和训练轮数可以提高模型质量
3. 对于生产环境，建议使用更大的模型和更多的训练数据

## 示例对话（你得先自己放入训练文件或者接入爬虫并进行训练）

启动聊天界面后，可以尝试以下对话：

```
你: 人工智能是什么？
AI: 人工智能是计算机科学的一个分支，致力于开发能够模拟和执行通常需要人类智能的任务的系统。这些任务包括视觉感知、语音识别、决策制定、自然语言处理等。

你: /set temperature 1.2
已将 temperature 设置为 1.2

你: 未来AI会如何发展？
AI: 未来人工智能技术可能会继续快速发展，向通用人工智能(AGI)方向迈进。可解释AI、低资源AI和人机协作将成为重要研究领域，同时安全性和伦理问题也将受到更多关注。
```

祝你使用愉快！如有任何问题，欢迎提问。 
