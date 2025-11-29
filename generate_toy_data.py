import os

def generate_data():
    os.makedirs("./data/toy", exist_ok=True)
    
    # 伪造 100 条简单的平行语料
    cn_sentences = [
        "你好", "今天天气很好", "我喜欢深度学习", "Transformer模型很强大", "我们需要更多的显卡",
        "时间就是金钱", "这是一个测试", "请把这句话翻译成英文", "人工智能改变世界", "编码器和解码器"
    ] * 10 # 复制多次以形成 Batch
    
    en_sentences = [
        "Hello", "The weather is good today", "I like deep learning", "Transformer model is powerful", "We need more GPUs",
        "Time is money", "This is a test", "Please translate this sentence to English", "AI changes the world", "Encoder and Decoder"
    ] * 10

    with open("./data/toy/train.cn", "w", encoding="utf-8") as f:
        for s in cn_sentences:
            f.write(s + "\n")
            
    with open("./data/toy/train.en", "w", encoding="utf-8") as f:
        for s in en_sentences:
            f.write(s + "\n")
            
    print("Toy dataset generated at ./data/toy/")

if __name__ == "__main__":
    generate_data()