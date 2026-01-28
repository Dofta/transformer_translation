# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import sacrebleu
from tqdm import tqdm
import math
import os
import warnings

warnings.filterwarnings("ignore")

from config import Config
from src.model import TransformerModel
from src.dataset import BilingualDataset
from src.utils import create_mask


def forbid_ngram_repeat(ys, logits, ngram_size=3):
    """
    Forcefully forbid generating repeated n-grams.
    If the previous n-1 tokens plus the potential next token form an n-gram
    that already exists in the generated sequence, forbid that token.
    """
    bs = ys.size(0)
    current_len = ys.size(1)
    if current_len < ngram_size:
        return logits

    for i in range(bs):
        # Get list of generated tokens
        gen_tokens = ys[i].tolist()

        # Current suffix (n-1) gram
        current_context = tuple(gen_tokens[-(ngram_size - 1) :])

        # Traverse history to see if this suffix appeared before
        # E.g. Generated: A B C D E [F G] ... prediction
        # Look for previous [F G]
        for step in range(len(gen_tokens) - (ngram_size - 1)):
            # Check sliding window
            previous_context = tuple(gen_tokens[step : step + (ngram_size - 1)])

            if previous_context == current_context:
                forbidden_token = gen_tokens[step + (ngram_size - 1)]
                logits[i, forbidden_token] = -float("inf")

    return logits


def compute_ppl(model, dataloader, device, pad_idx, criterion):
    """Calculate PPL"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating PPL"):
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask, tgt_mask, src_pad, tgt_pad = create_mask(
                src, tgt_input, pad_idx, device
            )

            # Enable BF16
            # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(src, tgt_input, src_mask, tgt_mask, src_pad, tgt_pad)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            valid_tokens = (tgt_out != pad_idx).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return ppl


def generate_translations(
    model, dataloader, tokenizer, device, raw_references, max_len=100, beam_size=4
):
    """
    Use Beam Search for decoding to completely solve the "never stop" problem.
    """
    model.eval()
    predictions = []

    sos_idx = tokenizer.token_to_id("[SOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")
    pad_idx = tokenizer.token_to_id("[PAD]")

    print(f"Generating with Beam Search (k={beam_size})...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoding"):
            src = batch["src_ids"].to(device)
            bs = src.size(0)

            # --- Encode once ---
            src_mask = src == pad_idx
            # Memory: [batch, src_len, d_model]
            memory = model.encode(src, src_mask=None, src_key_padding_mask=src_mask)

            # --- Beam Search for each Sample ---

            batch_preds = []

            for i in range(bs):
                # Extract current sample's memory: [1, src_len, d_model]
                curr_memory = memory[i].unsqueeze(0)
                curr_src_mask = src_mask[i].unsqueeze(0)

                # Beam: List of tuples (score, sequence_tensor)
                # Initial state: score=0.0, seq=[SOS]
                beams = [(0.0, torch.tensor([sos_idx], device=device))]

                finished_beams = []

                for _ in range(max_len):
                    new_beams = []

                    # Expand each beam
                    for score, seq in beams:
                        # Prepare input: [1, seq_len]
                        ys = seq.unsqueeze(0)

                        # Decode
                        out = model.decode(
                            ys, curr_memory, memory_key_padding_mask=curr_src_mask
                        )
                        logits = model.generator(out[:, -1])  # [1, vocab_size]

                        # Log Softmax (Use log probabilities for score accumulation)
                        log_probs = torch.log_softmax(logits, dim=1)  # [1, vocab]

                        # Take Top K candidates
                        # If first iteration, beams has 1 item, take k
                        # If subsequent, select k best from k*vocab, simplified here:
                        # Find best k for each beam, then summarize and select k
                        topk_probs, topk_ids = torch.topk(log_probs, beam_size)

                        for k in range(beam_size):
                            next_score = score + topk_probs[0, k].item()
                            next_word = topk_ids[0, k].item()

                            # Length Penalty: Lower or raise score for long sentences

                            new_seq = torch.cat(
                                [seq, torch.tensor([next_word], device=device)]
                            )

                            if next_word == eos_idx:
                                # Finished, add to finished list
                                # Normalize score (higher is better, but log_prob is negative, closer to 0 is better)
                                # Simple normalization: divide by length alpha=0.7
                                alpha = 0.7
                                norm_score = next_score / (new_seq.shape[0] ** alpha)
                                finished_beams.append((norm_score, new_seq))
                            else:
                                new_beams.append((next_score, new_seq))

                    # --- Prune: Keep the best k ---
                    # Sort by score in descending order (because log_prob, higher is better)
                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:beam_size]

                    # If all beams are finished, or reached a certain number
                    if len(finished_beams) >= beam_size:
                        break

                # Select the best result
                if len(finished_beams) > 0:
                    best_beam = max(finished_beams, key=lambda x: x[0])
                    final_seq = best_beam[1]
                else:
                    # If max_len reached without finish, select highest score current
                    final_seq = beams[0][1]

                # Decode to text
                ys_list = final_seq.tolist()

                # Remove Special Tokens
                clean_ids = [t for t in ys_list if t not in [sos_idx, eos_idx, pad_idx]]
                tokens = [
                    tokenizer.id_to_token(t)
                    for t in clean_ids
                    if tokenizer.id_to_token(t)
                ]

                # Assemble
                sent = " ".join(tokens).replace(" ##", "")
                batch_preds.append(sent)

            predictions.extend(batch_preds)

    return predictions, raw_references


def main():
    cfg = Config()
    device = torch.device("cpu")

    # Load Custom Tokenizer
    tokenizer_path = "./data/tokenizer_custom_32k.json"
    print(f"Loading Tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("[PAD]")

    # Prepare test data
    with open("./data/test/test.cn", "r", encoding="utf-8") as f:
        src_test = [line.strip() for line in f][:200]  # Take first 200 test samples
    with open("./data/test/test.en", "r", encoding="utf-8") as f:
        tgt_test = [line.strip() for line in f][:200]

    print(f"Evaluating on {len(src_test)} samples...")

    # Use the class from dataset.py (it now supports list input)
    test_dataset = BilingualDataset(
        src_test[:50], tgt_test[:50], tokenizer, cfg.max_len
    )

    # collate_fn needs to be imported from the dataset module
    from src.dataset import collate_fn
    from functools import partial

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_id=pad_idx),
    )

    # Load model
    model_path = "./checkpoints/sanity_final/model_restart_best.pt"
    # If best is not found, use latest
    if not os.path.exists(model_path):
        model_path = "./checkpoints/sanity_final/model_restart_latest.pt"

    print(f"Loading Model from {model_path}...")
    model = TransformerModel(
        vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
    )
    model.to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Compute metrics
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
    ppl = compute_ppl(model, test_loader, device, pad_idx, criterion)
    print(f"\nPerplexity (PPL): {ppl:.4f}")

    preds, refs = generate_translations(
        model, test_loader, tokenizer, device, raw_references=tgt_test
    )

    # Compute SacreBLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"ChrF Score: {chrf.score:.2f}")

    print("\n--- Translation Examples ---")
    for i in range(3):
        print(f"Ref : {refs[i]}")
        print(f"Pred: {preds[i]}")
        print("-" * 30)


if __name__ == "__main__":
    main()
