import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin
import re
import logging

class WebCrawler:
    def __init__(self, max_pages=100, delay=1):
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.texts = []
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def clean_text(self, text):
        """清理文本，去除HTML标签和多余空白"""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text.strip()
    
    def extract_text(self, soup):
        """从BeautifulSoup对象中提取文本"""
        # 移除脚本和样式元素
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取文本
        text = soup.get_text()
        return self.clean_text(text)
    
    def crawl(self, start_url, domain=None):
        """爬取网页内容"""
        if domain is None:
            domain = start_url.split('/')[2]
        
        try:
            response = requests.get(start_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取文本
            text = self.extract_text(soup)
            if text:
                self.texts.append(text)
                self.logger.info(f"已提取文本，长度: {len(text)}")
            
            # 提取链接
            links = soup.find_all('a', href=True)
            for link in links:
                url = urljoin(start_url, link['href'])
                if (url not in self.visited_urls and 
                    domain in url and 
                    len(self.texts) < self.max_pages):
                    self.visited_urls.add(url)
                    time.sleep(self.delay + random.random())  # 随机延迟
                    self.crawl(url, domain)
        
        except Exception as e:
            self.logger.error(f"爬取 {start_url} 时出错: {str(e)}")
    
    def get_training_data(self):
        """获取训练数据"""
        return self.texts
    
    def save_texts(self, filename):
        """保存爬取的文本到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(text + '\n')
        self.logger.info(f"文本已保存到 {filename}")
    
    @staticmethod
    def load_texts(filename):
        """从文件加载文本"""
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()] 