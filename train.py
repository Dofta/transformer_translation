# train.py

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from config import Config
from src.dataset import BilingualDataset, get_or_build_tokenizer, collate_fn
from src.model import TransformerModel
from src.utils import create_mask
from src.logger import TrainingLogger

# --- Scheduler ---
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def evaluate(model, dataloader, loss_fn, device, pad_idx, dtype, cfg):
    """Evaluation loop"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src_ids"].to(cfg.device)
            tgt = batch["tgt_ids"].to(cfg.device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(
                src, tgt_input, pad_idx, device
            )

            # Use BF16 context
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(
                    src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask
                )
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(override_config=None):
    # 5080 acceleration settings
    torch.set_float32_matmul_precision("high")

    cfg = Config()
    if override_config:
        for k, v in override_config.items():
            setattr(cfg, k, v)
    exp_name = (
        override_config.get("exp_name", "default_exp")
        if override_config
        else "default_exp"
    )
    logger = TrainingLogger(
        cfg.log_dir, model_name=f"tf_{cfg.mode}_{exp_name}"
    )  # Change name to distinguish

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    train_cn_path = os.path.join(cfg.data_dir, "train.cn")
    train_en_path = os.path.join(cfg.data_dir, "train.en")

    with open(train_cn_path, "r", encoding="utf-8") as f:
        src_train = [l.strip() for l in f]
    with open(train_en_path, "r", encoding="utf-8") as f:
        tgt_train = [l.strip() for l in f]

    # 3. Load validation data (use your generated test set as validation, or sanity set)
    if cfg.mode == "production":
        val_cn_path = os.path.join(cfg.test_data_dir, "test.cn")
        val_en_path = os.path.join(cfg.test_data_dir, "test.en")
    else:
        val_cn_path = train_cn_path
        val_en_path = train_en_path

    with open(val_cn_path, "r", encoding="utf-8") as f:
        src_val = [l.strip() for l in f]
    with open(val_en_path, "r", encoding="utf-8") as f:
        tgt_val = [l.strip() for l in f]

    tokenizer = get_or_build_tokenizer(cfg, src_train + tgt_train)
    pad_idx = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()

    # 5. Dataloaders
    train_dataset = BilingualDataset(src_train, tgt_train, tokenizer, cfg.max_len)
    val_dataset = BilingualDataset(src_val, tgt_val, tokenizer, cfg.max_len)

    # Use partial to fix pad_id parameter, so it can be pickled
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_idx),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=pad_idx),
        num_workers=0,
        pin_memory=True,
    )

    # --- Initialize model ---
    model = TransformerModel(
        vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
    ).to(cfg.device)

    # If you want to load weights but reset Epoch and Optimizer
    checkpoint_path = "./checkpoints/sanity/model_latest.pt"

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path} for RESTART...")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

        # Only load model parameters
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("Weights loaded. Optimizer RESET for aggressive training.")
    else:
        print("Warning: No checkpoint found. Starting from scratch.")

    # --- 5080 Compile Speedup (PyTorch 2.0) --- Only for Linux + 5080
    # print("Compiling model for RTX 5080...")
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"Compile failed: {e}")

    # --- Redefine optimizer (high LR) ---
    # Do not load optimizer from checkpoint, use new LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Use Cosine Annealing Warm Restarts strategy
    # T_0=5: restart every 5 epochs (let LR go down then bounce back)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_idx, label_smoothing=cfg.label_smoothing
    )

    # --- Training loop (with gradient accumulation) ---
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0

    print(
        f"Start RESTART training for {cfg.num_epochs} epochs with Accumulation={cfg.accumulation_steps}..."
    )

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=True
        )

        # Zero gradients once before the loop
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):

            # Extract data from dictionary by key
            src = batch["src_ids"].to(cfg.device)
            tgt = batch["tgt_ids"].to(cfg.device)

            # If you need masks later, you can also extract them (optional)
            # src_padding_mask = batch['attention_mask'].to(cfg.device)
            tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]

            src_mask, tgt_mask, src_pad, tgt_pad = create_mask(
                src, tgt_input, pad_idx, cfg.device
            )

            # --- BF16 Mixed Precision (No Scaler) ---
            with torch.autocast(device_type="cuda", dtype=cfg.dtype):
                logits = model(src, tgt_input, src_mask, tgt_mask, src_pad, tgt_pad)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

                # Gradient accumulation: divide loss by accumulation steps
                loss = loss / cfg.accumulation_steps

            # Backward (direct backpropagation)
            loss.backward()

            # --- Update only after accumulating N steps ---
            if (i + 1) % cfg.accumulation_steps == 0:
                # Gradient clipping (to prevent explosion due to large LR after Restart)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # Restore loss value for display
                current_loss = loss.item() * cfg.accumulation_steps
                total_train_loss += current_loss

                current_lr = optimizer.param_groups[0]["lr"]
                logger.log_step(global_step, current_loss, current_lr)

                progress_bar.set_postfix(
                    {"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.2e}"}
                )

        # --- End of Epoch ---
        # Update Scheduler
        scheduler.step()

        epoch_duration = time.time() - start_time
        # Note: Since total_loss is only accumulated on accumulate steps, divide by actual update count to get average
        avg_train_loss = total_train_loss / (len(train_loader) / cfg.accumulation_steps)

        # Validation
        print("Validating...")
        avg_val_loss = evaluate(
            model, val_loader, loss_fn, cfg.device, pad_idx, cfg.dtype, cfg
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(
            epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_duration
        )

        # Save logic, add timestamp to prevent overwrite
        checkpoint_dict = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg.__dict__,  # Save config as well
        }

        # Save latest
        torch.save(
            checkpoint_dict, os.path.join(cfg.model_save_dir, "model_restart_latest.pt")
        )

        if avg_val_loss < best_val_loss:
            print(f"New Best Val Loss! ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
            torch.save(
                checkpoint_dict,
                os.path.join(cfg.model_save_dir, "model_restart_best.pt"),
            )

    logger.close()
    print("Restart Training Complete!")


if __name__ == "__main__":
    train()
