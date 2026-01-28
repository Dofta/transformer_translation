import os
import torch


class Config:
    def __init__(self):
        # Mode selection: 'toy', 'sanity', 'production'
        self.mode = "sanity"

        # Path mapping
        self.data_map = {
            "toy": "./data/toy",
            "sanity": "./data/sanity",  # 100k real data pairs
            "production": "./data/bist_1.5m",  # 1.5 million real data pairs
        }
        self.data_dir = self.data_map[self.mode]

        # Automatically adjust training parameters
        if self.mode == "toy":
            self.batch_size = 16
            self.num_epochs = 5
            self.warmup_steps = 100
        elif self.mode == "sanity":
            self.batch_size = (
                96  # 5080 VRAM is large, sanity can also have a large batch
            )
            self.accumulation_steps = 1
            self.num_epochs = 30  # Quickly run to see convergence
            self.warmup_steps = 4000
        else:  # production
            self.batch_size = 96  # Adjusted based on 5080 VRAM
            self.accumulation_steps = (
                5  # Gradient accumulation, equivalent to a larger batch
            )
            self.num_epochs = 5
            self.warmup_steps = 1000

        self.accumulation_steps = 5 if self.mode == "production" else 1

        self.tokenizer_path = f"./data/tokenizer_{self.mode}.json"

        self.model_save_base = "./checkpoints"

        # Recommended to add timestamp to avoid overwriting (optional, depends on disk space)
        # self.run_id = time.strftime("%Y%m%d_%H%M")

        # If you want to overwrite old models under the same mode (save space, facilitate resuming), do not add specific time
        self.run_id = self.mode

        self.model_save_dir = os.path.join(self.model_save_base, self.run_id)

        # --- Data related ---
        self.src_lang = "cn"
        self.tgt_lang = "en"
        self.vocab_size = 5000 if self.mode == "toy" else 32000
        self.max_len = 64 if self.mode == "toy" else 128

        # --- Model parameters (Transformer Base) ---
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2048
        self.dropout = 0.3

        # --- Training parameters ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.label_smoothing = 0.1

        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        # Logging related
        self.log_dir = "./logs"
        self.log_interval = (
            100  # How many steps between logging step loss to TensorBoard
        )

        # Additional dataset paths
        # Assuming test dataset is also placed in data/test directory
        self.test_data_dir = "./data/test"
