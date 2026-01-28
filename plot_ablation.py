import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_experiments(logs_base_dir="./logs/exp_data"):
    # 1. Find metrics.csv for all experiments
    # Assuming directory structure is logs/tf_exp_name_timestamp/metrics.csv
    search_path = os.path.join(logs_base_dir, "*", "metrics.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print("No metrics.csv found! Please check your logs directory.")
        return

    data = {}
    
    print(f"Found {len(csv_files)} experiments.")

    # 2. Read data
    for f in csv_files:
        # Get experiment name (folder name without timestamp)
        dir_name = os.path.basename(os.path.dirname(f))
        parts = dir_name.split('_')
        if len(parts) > 3:
            exp_name = "_".join(parts[1:-2]) # Use middle part as legend
        else:
            exp_name = dir_name
            
        df = pd.read_csv(f)
        data[exp_name] = df

    # 3. Plot Training Loss comparison
    plt.figure(figsize=(10, 6))
    for name, df in data.items():
        plt.plot(df['epoch'], df['train_loss'], label=f"{name} (Train)", linestyle='-')
    
    plt.title("Ablation Study: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ablation_train_loss.png", dpi=300)
    print("Saved ablation_train_loss.png")

    # 4. Plot Validation Loss comparison (most important)
    plt.figure(figsize=(10, 6))
    for name, df in data.items():
        # Use dashed line for validation, or solid line is also fine
        plt.plot(df['epoch'], df['val_loss'], label=f"{name} (Val)", marker='.')
    
    plt.title("Ablation Study: Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ablation_val_loss.png", dpi=300)
    print("Saved ablation_val_loss.png")

    # 5. Generate summary table (Markdown format, ready for report)
    print("\n" + "="*50)
    print("Summary Table (Epoch 10 Comparison)")
    print("="*50)
    print(f"{'Experiment':<25} | {'Train Loss':<10} | {'Val Loss':<10} | {'PPL (Est.)':<10}")
    print("-" * 65)
    
    import math
    for name, df in data.items():
        # Try to get data from epoch 10 (index 9), if not enough then use last epoch
        target_idx = 9 if len(df) >= 10 else -1
        row = df.iloc[target_idx]
        epoch_num = int(row['epoch'])
        
        # Estimate PPL = exp(loss)
        ppl = math.exp(row['val_loss']) if row['val_loss'] < 100 else float('inf')
        
        print(f"{name:<25} | {row['train_loss']:.4f}     | {row['val_loss']:.4f}     | {ppl:.2f}")
        
    print("-" * 65)
    print("Note: Comparison based on Epoch 10 (or last epoch if <10).")

if __name__ == "__main__":
    # Make sure this path points to logs folder
    plot_experiments("./logs/exp_data")