import torch
import os
from tokenizers import Tokenizer
import re
import warnings

warnings.filterwarnings("ignore")

from config import Config
from src.model import TransformerModel

# --- Configuration ---
MODEL_PATH = "./checkpoints/sanity_final/model_restart_best.pt"  # Path to the trained model weights
TOKENIZER_PATH = "./data/tokenizer_custom_32k.json"  # Path to 32k tokenizer
BEAM_SIZE = 4  # Search width
ALPHA = 1.0  # Length penalty (1.0=neutral, >1.0=encourage longer sentences)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stuff():
    """Load configuration, tokenizer, and model"""
    print(f"Initializing on {DEVICE}...")
    cfg = Config()

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded (Vocab: {vocab_size})")

    # Initialize model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_head=cfg.n_heads,
        n_layer=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=0.0,  # Dropout not needed for inference
        max_len=5000,
    )

    # Load weights
    if not os.path.exists(MODEL_PATH):
        # Try to find latest
        latest_path = MODEL_PATH.replace("best", "latest")
        if os.path.exists(latest_path):
            print(f"Best model not found, loading latest: {latest_path}")
            checkpoint_path = latest_path
        else:
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    else:
        checkpoint_path = MODEL_PATH

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Compatible with different save formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    return model, tokenizer, cfg


def translate_beam_search(text, model, tokenizer, config):
    """
    Use Beam Search for high-quality translation
    """
    # Preprocessing
    # Clean input by stripping leading and trailing spaces
    text = text.strip()
    encoded = tokenizer.encode(text)
    src_ids = torch.tensor(encoded.ids).unsqueeze(0).to(DEVICE)  # [1, seq_len]

    sos_idx = tokenizer.token_to_id("[SOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")
    pad_idx = tokenizer.token_to_id("[PAD]")

    # Encode
    src_mask = src_ids == pad_idx
    with torch.no_grad():
        # Use float32 for inference to ensure compatibility and precision
        memory = model.encode(src_ids, src_mask=None, src_key_padding_mask=src_mask)

    # Beam Search initialization
    # beams: list of (score, sequence_tensor)
    beams = [(0.0, torch.tensor([sos_idx], device=DEVICE))]
    finished_beams = []

    for _ in range(config.max_len):
        new_beams = []

        for score, seq in beams:
            # Prepare input: [1, curr_len]
            ys = seq.unsqueeze(0)

            with torch.no_grad():
                out = model.decode(ys, memory, memory_key_padding_mask=src_mask)
                logits = model.generator(out[:, -1])

                # N-Gram Blocking (Prevent consecutive 3-gram repetition)
                # Simple version: prevent repeating the last token
                last_token = seq[-1].item()
                logits[0, last_token] -= 10.0  # Soft penalty

                log_probs = torch.log_softmax(logits, dim=1)

            # Get Top K
            topk_probs, topk_ids = torch.topk(log_probs, BEAM_SIZE)

            for k in range(BEAM_SIZE):
                next_score = score + topk_probs[0, k].item()
                next_word = topk_ids[0, k].item()
                new_seq = torch.cat([seq, torch.tensor([next_word], device=DEVICE)])

                if next_word == eos_idx:
                    # Length normalization: divide by length raised to the power of alpha
                    length_penalty = new_seq.shape[0] ** ALPHA
                    final_score = next_score / length_penalty
                    finished_beams.append((final_score, new_seq))
                else:
                    new_beams.append((next_score, new_seq))

        # Prune: keep the best K
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:BEAM_SIZE]

        # If enough finished sentences are collected
        if len(finished_beams) >= BEAM_SIZE:
            break

    # Choose the best result
    if not finished_beams:
        finished_beams = beams  # If no EOS generated, use current sequences

    best_beam = max(finished_beams, key=lambda x: x[0])
    final_seq = best_beam[1].tolist()

    # Decoding and post-processing
    # Filter out special tokens
    clean_ids = [t for t in final_seq if t not in [sos_idx, eos_idx, pad_idx]]

    # Manually join tokens to fix spacing issues caused by BertPreTokenizer
    tokens = [tokenizer.id_to_token(t) for t in clean_ids if tokenizer.id_to_token(t)]
    result_text = " ".join(tokens)

    # Clean up artifacts possibly caused by BPE
    result_text = result_text.replace(" ##", "").replace(" .", ".").replace(" ,", ",")

    return result_text


if __name__ == "__main__":
    try:
        model, tokenizer, cfg = load_stuff()

        print("\n" + "=" * 50)
        print("Transformer Translator (Interactive Mode)")
        print(f"Beam Size: {BEAM_SIZE} | Length Penalty: {ALPHA}")
        print("Input 'q' or 'exit' to quit")
        print("=" * 50 + "\n")

        while True:
            text = input("请输入中文句子 (Input CN): ")
            if text.lower() in ["q", "exit"]:
                break

            if not text.strip():
                continue

            print("⏳ Translating...", end="\r")
            try:
                translation = translate_beam_search(text, model, tokenizer, cfg)
                print(f"🇺🇸 Result: \033[92m{translation}\033[0m")  # Green highlight
                print("-" * 30)
            except Exception as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        print(f"\nInitialization Failed: {e}")
