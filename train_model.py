from ai_meta_model import AIMetaModel
import matplotlib.pyplot as plt
import sys

def main():
    print("程序开始执行...")
    
    # 示例训练数据
    training_texts = [
        "人工智能正在改变我们的生活方式",
        "深度学习模型在自然语言处理领域取得了突破性进展",
        "大语言模型如GPT和BERT能够生成人类难以区分的文本",
        "向量化表示是现代自然语言处理的基础",
        "概率模型可以计算词与词之间的转移概率",
        "AI元模型通过分析文本来学习语言的结构",
        "机器学习算法需要大量数据来训练",
        "神经网络是深度学习的核心组件",
        "词向量能够捕捉词语之间的语义关系",
        "自然语言处理技术使得机器能够理解人类语言"
    ]
    
    print(f"准备训练数据，共{len(training_texts)}条文本")
    
    try:
        # 初始化模型
        print("\n初始化AI元模型...")
        model = AIMetaModel(vector_size=50, window_size=2, min_count=1)
        
        # 训练模型
        print("开始训练模型...")
        model.train(training_texts)
        
        # 生成文本
        print("\n生成一些文本示例:")
        for start_word in ["人工智能", "深度学习", "模型"]:
            if start_word in model.vocab:
                generated = model.generate_text(start_word, length=8)
                print(f"以'{start_word}'开始: {generated}")
            else:
                print(f"警告：'{start_word}'不在词汇表中")
        
        # 可视化词向量拓扑
        print("\n生成词向量拓扑结构可视化...")
        model.visualize_topology(n_components=3)
        
        # 保存模型
        print("\n保存模型...")
        model.save("ai_meta_model.pkl")
        
        # 加载并测试模型
        print("\n加载模型并测试...")
        loaded_model = AIMetaModel.load("ai_meta_model.pkl")
        print("使用加载的模型生成文本:")
        generated = loaded_model.generate_text("自然语言", length=8)
        print(f"生成的文本: {generated}")
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 