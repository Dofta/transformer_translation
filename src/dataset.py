# src/dataset.py

import torch
import os
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers


class BilingualDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, tokenizer, max_len):
        self.max_len = max_len
        self.pad_idx = tokenizer.token_to_id("[PAD]")

        print(f"Sanitizing & Pre-tokenizing {len(src_sentences)} samples...")

        self.src_data = []
        self.tgt_data = []

        # Batch Encoding (Fastest method)
        # First clean
        clean_src = [str(s).strip() for s in src_sentences if s and str(s).strip()]
        clean_tgt = [str(t).strip() for t in tgt_sentences if t and str(t).strip()]

        # Truncate longer list to match (prevent some lines having src but no tgt)
        min_len = min(len(clean_src), len(clean_tgt))
        clean_src = clean_src[:min_len]
        clean_tgt = clean_tgt[:min_len]

        # Batch Encode
        # enable_padding=False (to save memory, padding done in collate)
        # truncation=True
        enc_src = tokenizer.encode_batch(clean_src)
        enc_tgt = tokenizer.encode_batch(clean_tgt)

        for src_out, tgt_out in zip(enc_src, enc_tgt):
            # Manual truncation (Although Tokenizer has truncation parameter, sometimes explicit configuration is needed in encode_batch)
            # Here simply take ids and cut off if too long
            s_ids = src_out.ids[:max_len]
            t_ids = tgt_out.ids[:max_len]
            self.src_data.append(s_ids)
            self.tgt_data.append(t_ids)

        print(f"Pre-tokenization complete! Valid pairs: {len(self.src_data)}")

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {"src_ids": self.src_data[idx], "tgt_ids": self.tgt_data[idx]}


def get_or_build_tokenizer(config, sentences_list):
    """
    Train a BPE Tokenizer optimized for Chinese (32k)
    """
    tokenizer_path = "./data/tokenizer_custom_32k.json"

    # If local exists, load directly
    if os.path.exists(tokenizer_path):
        print(f"📂 Loading custom tokenizer from {tokenizer_path}...")
        return Tokenizer.from_file(tokenizer_path)

    # If not, train on the spot (only takes a few seconds)
    print("Training NEW custom tokenizer (BertPreTokenizer + BPE)...")

    # Use BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Use BertPreTokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # Decoder
    tokenizer.decoder = (
        decoders.ByteLevel()
    )  # or WordPiece, here ByteLevel is more general

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=32000,  # Vocabulary size controlled at 32k
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        show_progress=True,
    )

    # Post-processor (automatically add SOS/EOS)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SOS] $A [EOS]",
        pair="[SOS] $A [EOS] [SOS] $B [EOS]",
        special_tokens=[
            (
                "[SOS]",
                2,
            ),  # Assume ID allocation order, will auto-align after training, just a placeholder here
            ("[EOS]", 3),
        ],
    )

    # Start training
    # sentences_list is a generator or list containing all Chinese and English sentences
    tokenizer.train_from_iterator(sentences_list, trainer=trainer)

    # Save
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer


def collate_fn(batch, pad_id):
    src_ids = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_ids = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_ids, batch_first=True, padding_value=pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_ids, batch_first=True, padding_value=pad_id
    )

    return {
        "src_ids": src_padded,
        "tgt_ids": tgt_padded,
        "attention_mask": (src_padded != pad_id),
    }
