from auto_trainer import AutoTrainer
import sys
import os
import threading
import time

class ChatInterface:
    def __init__(self, model_path="trained_model.pkl"):
        self.trainer = AutoTrainer(model_path=model_path)
        self.is_running = True
        self.update_thread = None
    
    def start_auto_update(self, start_url, update_interval=3600):
        """启动自动更新线程"""
        self.update_thread = threading.Thread(
            target=self.trainer.auto_update,
            args=(start_url, update_interval),
            daemon=True
        )
        self.update_thread.start()
        print(f"已启动自动更新线程，每 {update_interval} 秒更新一次")
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*50)
        print("欢迎使用AI对话系统")
        print("="*50)
        print("\n命令列表：")
        print("1. /help - 显示帮助信息")
        print("2. /train - 手动触发训练")
        print("3. /quit - 退出程序")
        print("4. 直接输入文本进行对话")
        print("\n" + "-"*50 + "\n")
    
    def handle_command(self, command):
        """处理特殊命令"""
        if command == "/help":
            self.print_welcome()
        elif command == "/train":
            print("开始训练模型...")
            if self.trainer.train_model():
                print("训练成功！")
            else:
                print("训练失败，请检查日志文件。")
        elif command == "/quit":
            self.is_running = False
            print("正在退出程序...")
        else:
            print("未知命令，输入 /help 查看帮助")
    
    def chat(self):
        """开始对话"""
        self.print_welcome()
        
        while self.is_running:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                # 处理特殊命令
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue
                
                # 生成回复
                response = self.trainer.generate_response(user_input)
                print("\nAI:", response)
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                self.is_running = False
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
                print("输入 /help 查看帮助，或 /quit 退出程序")
        
        print("感谢使用，再见！")

def main():
    # 创建聊天界面
    chat = ChatInterface()
    
    # 设置自动更新的起始URL（这里使用知乎作为示例）
    start_url = "https://www.zhihu.com/topic/19550517/hot"
    
    # 启动自动更新线程（每小时更新一次）
    chat.start_auto_update(start_url, update_interval=3600)
    
    # 开始对话
    chat.chat()

if __name__ == "__main__":
    main() 