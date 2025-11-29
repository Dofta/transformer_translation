import torch

class Config:
    def __init__(self):
        # --- 模式开关 ---
        self.debug_mode = True  # True: 使用 Toy 数据快速验证流程; False: 使用正式数据
        
        # --- 路径配置 ---
        self.data_dir = "./data/toy" if self.debug_mode else "./data/bist"
        self.tokenizer_path = "./data/tokenizer.json"
        self.model_save_dir = "./checkpoints"
        
        # --- 数据相关 ---
        self.src_lang = "cn"
        self.tgt_lang = "en"
        # 验证模式下词表设小一点，正式训练建议 32000-37000
        self.vocab_size = 5000 if self.debug_mode else 32000 
        self.max_len = 64 if self.debug_mode else 128
        
        # --- 模型参数 (Transformer Base) ---
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1
        
        # --- 训练参数 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16 if self.debug_mode else 128  # 5080可以尝试更大
        self.num_epochs = 5 if self.debug_mode else 20
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.label_smoothing = 0.1
        self.warmup_steps = 100 if self.debug_mode else 4000
        
        # 混合精度训练 (5080 必备)
        self.use_amp = True

        # 日志相关
        self.log_dir = "./logs"
        self.log_interval = 100  # 每多少步记录一次 step loss 到 TensorBoard
        
        # 数据集路径补充
        # 假设 test 数据集也放在 data/test 目录下
        self.test_data_dir = "./data/test"
        