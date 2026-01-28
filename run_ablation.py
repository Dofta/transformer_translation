import torch
from train_new import train
import gc

# Define the ablation study groups
# Assuming Base Config is: n_layers=6, dropout=0.1, tied_embeddings=True
experiments = [
    # Experiment 1: Baseline
    {
        "exp_name": "baseline_100k",
        "dropout": 0.3,  # Recommended 0.3 for small datasets
        "n_layers": 6,
        "n_heads": 8
    },
    # Experiment 2: Reduce number of layers (to see if shallow models are more friendly to small datasets)
    {
        "exp_name": "ablation_3layers",  # 3 layers
        "dropout": 0.3,
        "n_layers": 3, 
        "n_heads": 8,
        "vocab_size": 32000,
        "batch_size": 128,  # Fewer layers use less VRAM, so batch size can be increased for faster training
        "num_epochs": 10   # Quick validation
    },
    
    # Experiment 3: Low Dropout (to check underfitting/overfitting)
    {
        "exp_name": "ablation_dropout_0.1",  # Weak regularization
        "dropout": 0.1, 
        "n_layers": 6,
        "n_heads": 8,
        "vocab_size": 32000,
        "batch_size": 96,
        "num_epochs": 10   # Quick validation
    },
    
    # Experiment 4: Fewer heads (to verify the necessity of multi-head attention)
    {
        "exp_name": "ablation_4heads",
        "dropout": 0.3,
        "n_layers": 6,
        "n_heads": 4, 
        "vocab_size": 32000,
        "batch_size": 96,
        "num_epochs": 10   # Quick validation
    }
]

if __name__ == "__main__":
    for i, exp_config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Running Experiment {i+1}/{len(experiments)}: {exp_config['exp_name']}")
        print(f"Epochs: {exp_config['num_epochs']}")
        print(f"{'='*60}\n")
        
        try:
            train(override_config=exp_config)
        except Exception as e:
            print(f"Experiment failed: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print("\nAll Ablation Studies Completed!")