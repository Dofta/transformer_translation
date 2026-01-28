import os
import time
import csv
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir, model_name="transformer"):
        # Create a log directory with timestamp to prevent overwriting previous experiments
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
        # CSV Writer (for recording summary data of each epoch)
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.csv_header = ["epoch", "train_loss", "val_loss", "learning_rate", "epoch_time_sec"]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
            
        print(f"Logging to: {self.run_dir}")

    def log_step(self, step, train_loss, lr):
        """Log training information for each step (TensorBoard)"""
        self.writer.add_scalar("Train/Loss_Step", train_loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)

    def log_epoch(self, epoch, train_loss, val_loss, lr, duration):
        """Log summary information for each epoch (TensorBoard + CSV)"""
        # TensorBoard
        self.writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
        self.writer.add_scalar("Val/Loss_Epoch", val_loss, epoch)
        
        # CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr, round(duration, 2)])
        
        # Print to console
        print(f"Epoch {epoch} | Time: {duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")

    def close(self):
        self.writer.close()