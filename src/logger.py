import os
import time
import csv
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir, model_name="transformer"):
        # 创建带有时间戳的日志目录，防止覆盖之前的实验
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 1. TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
        # 2. CSV Writer (用于记录每个Epoch的汇总数据)
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.csv_header = ["epoch", "train_loss", "val_loss", "learning_rate", "epoch_time_sec"]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
            
        print(f"Logging to: {self.run_dir}")

    def log_step(self, step, train_loss, lr):
        """记录每一步的训练信息 (TensorBoard)"""
        self.writer.add_scalar("Train/Loss_Step", train_loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)

    def log_epoch(self, epoch, train_loss, val_loss, lr, duration):
        """记录每个Epoch的汇总信息 (TensorBoard + CSV)"""
        # TensorBoard
        self.writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
        self.writer.add_scalar("Val/Loss_Epoch", val_loss, epoch)
        
        # CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr, round(duration, 2)])
        
        # 打印到控制台
        print(f"Epoch {epoch} | Time: {duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")

    def close(self):
        self.writer.close()