from ai_meta_model_advanced import TokenizerCN, AIMetaModelAdvanced, ModelTrainer
import argparse
import os
import logging
import torch
import time
import threading
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedChatInterface:
    def __init__(self, model_dir):
        """初始化聊天界面"""
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载分词器
        tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
        self.tokenizer = TokenizerCN.load(tokenizer_path)
        
        # 创建模型
        self.model = AIMetaModelAdvanced(
            vocab_size=len(self.tokenizer.word2idx),
            embed_dim=64,        # 减小嵌入维度
            num_heads=4,         # 减少注意力头数
            ff_dim=256,          # 减小前馈网络维度
            num_layers=2,        # 减少Transformer层数
            max_length=50,       # 减小最大序列长度
            dropout=0.1          # 减小dropout
        ).to(self.device)
        
        # 加载模型参数
        model_path = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 设置生成参数
        self.max_length = 200     # 增加最大生成长度
        self.temperature = 0.8    # 增加温度以提高创造性
        self.top_k = 40          # 减小top_k以增加连贯性
        self.top_p = 0.95        # 增加top_p以保持多样性
        
    def generate_response(self, prompt):
        """生成回复"""
        try:
            # 对输入进行分词
            tokens = self.tokenizer.tokenize(prompt)
            if not tokens:
                return "抱歉，我无法理解您的输入。"
            
            # 生成回复
            response = self.model.generate_text(
                self.tokenizer,
                prompt,  # 使用完整的用户输入
                max_length=self.max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            
            return response
            
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}", exc_info=True)
            return "抱歉，生成回复时出现错误。"
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*50)
        print("欢迎使用改进版AI元模型聊天系统！")
        print("="*50)
        print("\n可用命令：")
        print("/help - 显示帮助信息")
        print("/params - 显示当前生成参数")
        print("/set - 设置生成参数")
        print("/quit - 退出程序")
        print("\n开始聊天吧！输入 /help 查看详细说明。")
        print("-"*50 + "\n")
    
    def handle_command(self, command):
        """处理特殊命令"""
        if command == "/help":
            print("\n命令说明：")
            print("/help - 显示此帮助信息")
            print("/params - 显示当前生成参数")
            print("/set - 设置生成参数，格式：/set 参数名 值")
            print("  可用参数：max_length, temperature, top_k, top_p")
            print("/quit - 退出程序")
            return True
            
        elif command == "/params":
            print("\n当前生成参数：")
            print(f"max_length: {self.max_length}")
            print(f"temperature: {self.temperature}")
            print(f"top_k: {self.top_k}")
            print(f"top_p: {self.top_p}")
            return True
            
        elif command.startswith("/set"):
            try:
                _, param, value = command.split()
                value = float(value)
                
                if param == "max_length":
                    self.max_length = int(value)
                elif param == "temperature":
                    self.temperature = value
                elif param == "top_k":
                    self.top_k = int(value)
                elif param == "top_p":
                    self.top_p = value
                else:
                    print(f"未知参数：{param}")
                    return True
                    
                print(f"已将 {param} 设置为 {value}")
                return True
                
            except ValueError:
                print("参数设置格式错误，请使用：/set 参数名 值")
                return True
                
        elif command == "/quit":
            return False
            
        return True
    
    def chat(self):
        """主聊天循环"""
        self.print_welcome()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                # 处理特殊命令
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # 生成回复
                start_time = time.time()
                response = self.generate_response(user_input)
                generation_time = time.time() - start_time
                
                # 打印回复
                print(f"\nAI: {response}")
                print(f"\n生成用时: {generation_time:.2f}秒")
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                logger.error(f"聊天过程中出错: {str(e)}", exc_info=True)
                print("\n发生错误，请重试或输入 /quit 退出")
        
        print("\n感谢使用！再见！")

def create_model(vocab_size):
    """创建模型"""
    model = AIMetaModelAdvanced(
        vocab_size=vocab_size,
        embed_dim=256,      # 增加嵌入维度
        num_heads=8,        # 增加注意力头数
        ff_dim=1024,        # 增加前馈网络维度
        num_layers=8,       # 增加层数
        max_length=100,     # 保持最大长度
        dropout=0.2         # 增加dropout
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="改进版AI元模型聊天界面")
    parser.add_argument("--model_dir", type=str, default="./advanced_model", help="模型目录")
    args = parser.parse_args()
    
    try:
        chat_interface = AdvancedChatInterface(args.model_dir)
        chat_interface.chat()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 