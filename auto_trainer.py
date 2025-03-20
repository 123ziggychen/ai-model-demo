from ai_meta_model import AIMetaModel
from web_crawler import WebCrawler
import logging
import os
import time
from datetime import datetime

class AutoTrainer:
    def __init__(self, model_path="trained_model.pkl", data_path="training_data.txt"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def crawl_and_save(self, start_url, max_pages=100):
        """爬取网页并保存训练数据"""
        self.logger.info(f"开始从 {start_url} 爬取数据...")
        crawler = WebCrawler(max_pages=max_pages)
        crawler.crawl(start_url)
        crawler.save_texts(self.data_path)
        self.logger.info(f"数据已保存到 {self.data_path}")
    
    def train_model(self, vector_size=100, window_size=2, min_count=2):
        """训练模型"""
        try:
            # 加载训练数据
            if not os.path.exists(self.data_path):
                self.logger.error(f"训练数据文件 {self.data_path} 不存在")
                return False
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                self.logger.error("没有找到有效的训练数据")
                return False
            
            self.logger.info(f"加载了 {len(texts)} 条训练数据")
            
            # 初始化并训练模型
            self.model = AIMetaModel(
                vector_size=vector_size,
                window_size=window_size,
                min_count=min_count
            )
            
            self.model.train(texts)
            
            # 保存模型
            self.model.save(self.model_path)
            self.logger.info(f"模型已保存到 {self.model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练过程中出错: {str(e)}")
            return False
    
    def load_model(self):
        """加载已训练的模型"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"模型文件 {self.model_path} 不存在")
                return False
            
            self.model = AIMetaModel.load(self.model_path)
            self.logger.info("模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            return False
    
    def generate_response(self, prompt, max_length=50):
        """生成回复"""
        if not self.model:
            if not self.load_model():
                return "错误：模型未加载"
        
        try:
            # 将输入文本分词
            tokens = self.model.tokenize(prompt)
            if not tokens:
                return "请输入有效的文本"
            
            # 使用最后一个token作为起始点
            start_token = tokens[-1]
            response = self.model.generate_text(start_token, length=max_length)
            return response
            
        except Exception as e:
            self.logger.error(f"生成回复时出错: {str(e)}")
            return f"生成回复时出错: {str(e)}"
    
    def auto_update(self, start_url, update_interval=3600):
        """自动更新模型"""
        while True:
            try:
                # 爬取新数据
                self.crawl_and_save(start_url)
                
                # 训练新模型
                if self.train_model():
                    self.logger.info("模型更新成功")
                else:
                    self.logger.error("模型更新失败")
                
                # 等待下一次更新
                self.logger.info(f"等待 {update_interval} 秒后进行下一次更新...")
                time.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"自动更新过程中出错: {str(e)}")
                time.sleep(60)  # 出错后等待1分钟再试 