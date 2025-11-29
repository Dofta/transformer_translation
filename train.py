import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import Config
from src.dataset import BilingualDataset, get_or_build_tokenizer, collate_fn
from src.model import TransformerModel
from src.utils import create_mask
from src.logger import TrainingLogger # 新增

def evaluate(model, dataloader, loss_fn, device, pad_idx, amp_enabled):
    """验证循环：只计算Loss，不反向传播"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, pad_idx, device)
            
            # 使用与训练相同的混合精度上下文，虽然不做反向传播，但能保持一致性并加速
            with autocast(enabled=amp_enabled):
                logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def train():
    cfg = Config()
    
    # 1. 初始化 Logger
    logger = TrainingLogger(cfg.log_dir, model_name=f"tf_{cfg.mode}")
    
    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    # 2. 加载训练数据
    train_cn_path = os.path.join(cfg.data_dir, "train.cn")
    train_en_path = os.path.join(cfg.data_dir, "train.en")
    
    with open(train_cn_path, 'r', encoding='utf-8') as f: src_train = [l.strip() for l in f]
    with open(train_en_path, 'r', encoding='utf-8') as f: tgt_train = [l.strip() for l in f]

    # 3. 加载验证数据 (使用你生成的 test 集作为验证集，或者 sanity 集)
    # 建议：如果是 Production 模式，用 test 集做验证。如果是 Sanity 模式，用 Sanity 集做验证。
    if cfg.mode == 'production':
        val_cn_path = os.path.join(cfg.test_data_dir, "test.cn")
        val_en_path = os.path.join(cfg.test_data_dir, "test.en")
    else:
        # Sanity 模式下，验证集和训练集一样，仅为了测试代码流程
        val_cn_path = train_cn_path
        val_en_path = train_en_path
        
    with open(val_cn_path, 'r', encoding='utf-8') as f: src_val = [l.strip() for l in f]
    with open(val_en_path, 'r', encoding='utf-8') as f: tgt_val = [l.strip() for l in f]

    # 4. Tokenizer
    # 注意：Tokenizer 必须用所有数据训练，或者只用训练集训练（通常只用训练集）
    # 这里我们只用训练集构建 Tokenizer
    tokenizer = get_or_build_tokenizer(cfg, src_train + tgt_train)
    pad_idx = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()

    # 5. Dataloaders
    train_dataset = BilingualDataset(src_train, tgt_train, tokenizer, cfg.max_len)
    val_dataset = BilingualDataset(src_val, tgt_val, tokenizer, cfg.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                              collate_fn=lambda x: collate_fn(x, pad_idx), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, pad_idx), num_workers=4, pin_memory=True)

    # 6. Model
    model = TransformerModel(vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout).to(cfg.device)

    # 7. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader), epochs=cfg.num_epochs, pct_start=0.1
    ) # 推荐：使用 OneCycleLR 或 Transformer 专用的 Warmup scheduler
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg.label_smoothing)
    scaler = GradScaler(enabled=cfg.use_amp)

    # 8. Training Loop
    global_step = 0
    print(f"Start training for {cfg.num_epochs} epochs...")
    
    for epoch in range(cfg.num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(cfg.device), tgt.to(cfg.device)
            tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            src_mask, tgt_mask, src_pad, tgt_pad = create_mask(src, tgt_input, pad_idx, cfg.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=cfg.use_amp):
                logits = model(src, tgt_input, src_mask, tgt_mask, src_pad, tgt_pad)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # --- Log Step ---
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_step(global_step, loss.item(), current_lr)
            total_train_loss += loss.item()
            global_step += 1
            
            if i % cfg.log_interval == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # --- End of Epoch ---
        epoch_duration = time.time() - start_time
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        print("Validating...")
        avg_val_loss = evaluate(model, val_loader, loss_fn, cfg.device, pad_idx, cfg.use_amp)
        
        # Log Epoch
        logger.log_epoch(epoch+1, avg_train_loss, avg_val_loss, optimizer.param_groups[0]['lr'], epoch_duration)
        
        # Save Model (只保存最佳或最新的)
        # 这里简单起见，每个 Epoch 都存覆盖式 checkpoint，或者按 Epoch 存
        torch.save(model.state_dict(), f"{cfg.model_save_dir}/model_{cfg.mode}_latest.pt")
        # 也可以加逻辑：如果 val_loss 创新低，则保存 model_best.pt

    logger.close()
    print("Training Complete!")

if __name__ == "__main__":
    train()