from ai_meta_model_advanced import TokenizerCN, AIMetaModelAdvanced, ModelTrainer
from web_crawler import WebCrawler
import argparse
import os
import logging
import torch
import time
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_or_create_data(data_path, url=None, max_pages=50):
    """加载或创建训练数据"""
    if os.path.exists(data_path):
        logger.info(f"从 {data_path} 加载现有数据")
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    
    if url:
        logger.info(f"从 {url} 爬取数据")
        crawler = WebCrawler(max_pages=max_pages)
        crawler.crawl(url)
        crawler.save_texts(data_path)
        return crawler.get_training_data()
    
    raise ValueError("数据文件不存在且未提供URL")

def train_model(args):
    """训练模型"""
    # 加载数据
    texts = load_or_create_data(args.data_path, args.url, args.max_pages)
    logger.info(f"加载了 {len(texts)} 条文本进行训练")
    
    # 准备目录
    os.makedirs(args.model_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.pkl")
    model_path = os.path.join(args.model_dir, "model.pt")
    
    # 训练或加载分词器
    if os.path.exists(tokenizer_path) and not args.retrain_tokenizer:
        logger.info(f"从 {tokenizer_path} 加载分词器")
        tokenizer = TokenizerCN.load(tokenizer_path)
    else:
        logger.info("训练新的分词器")
        tokenizer = TokenizerCN(vocab_size=args.vocab_size)
        tokenizer.fit(texts)
        tokenizer.save(tokenizer_path)
    
    # 创建或加载模型
    if os.path.exists(model_path) and args.continue_training:
        logger.info(f"从 {model_path} 加载模型并继续训练")
        trainer = ModelTrainer.load_model(model_path, tokenizer)
    else:
        logger.info("创建新模型")
        model = AIMetaModelAdvanced(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=256,
            num_heads=8,
            ff_dim=1024,
            num_layers=8,
            max_length=100,
            dropout=0.2
        )
        trainer = ModelTrainer(model, tokenizer)
    
    # 训练模型
    start_time = time.time()
    logger.info("开始训练模型...")
    trainer.train(
        texts=texts,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        save_path=model_path
    )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成！用时 {training_time:.2f} 秒")
    
    # 评估模型
    logger.info("评估模型...")
    metrics = trainer.evaluate(texts, batch_size=args.batch_size)
    logger.info(f"评估结果: 损失 = {metrics['loss']:.4f}, 困惑度 = {metrics['perplexity']:.4f}")
    
    # 可视化嵌入
    if args.visualize:
        logger.info("可视化词嵌入...")
        trainer.visualize_embeddings()
    
    return trainer

def generate_samples(trainer, n_samples=5):
    """生成一些文本样本"""
    logger.info(f"生成 {n_samples} 个文本样本:")
    for i in range(n_samples):
        # 随机选择一个起始词
        start_idx = torch.randint(4, len(trainer.tokenizer.word2idx), (1,)).item()
        start_token = trainer.tokenizer.idx2word[start_idx]
        
        # 生成文本
        generated = trainer.model.generate_text(
            trainer.tokenizer, 
            start_token, 
            max_length=50,
            temperature=0.7
        )
        
        logger.info(f"样本 {i+1} (起始词: {start_token}): {generated}")

def create_model(vocab_size):
    """创建模型"""
    model = AIMetaModelAdvanced(
        vocab_size=vocab_size,
        embed_dim=128,      # 保持原有维度
        num_heads=4,        # 保持原有头数
        ff_dim=512,         # 保持原有前馈网络维度
        num_layers=4,       # 保持原有层数
        max_length=100,     # 保持原有最大长度
        dropout=0.1         # 保持原有dropout
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="训练改进版AI元模型")
    parser.add_argument("--data_path", type=str, default="advanced_training_data.txt", help="训练数据路径")
    parser.add_argument("--url", type=str, default=None, help="爬取数据的URL")
    parser.add_argument("--max_pages", type=int, default=50, help="最大爬取页数")
    parser.add_argument("--model_dir", type=str, default="./advanced_model", help="模型保存目录")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词汇表大小")
    parser.add_argument("--embed_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--ff_dim", type=int, default=512, help="前馈网络维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--max_length", type=int, default=100, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--continue_training", action="store_true", help="继续训练现有模型")
    parser.add_argument("--retrain_tokenizer", action="store_true", help="重新训练分词器")
    parser.add_argument("--visualize", action="store_true", help="可视化词嵌入")
    parser.add_argument("--generate", action="store_true", help="生成文本样本")
    
    args = parser.parse_args()
    
    try:
        # 训练模型
        trainer = train_model(args)
        
        # 生成样本
        if args.generate:
            generate_samples(trainer)
            
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 