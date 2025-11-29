import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class BilingualDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, tokenizer, max_len):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        # 编码
        enc_src = self.tokenizer.encode(src_text)
        enc_tgt = self.tokenizer.encode(tgt_text)

        # 截断与Padding将在 collate_fn 中处理，或者在这里处理
        # 为简单起见，这里直接截断，Padding 留给 Collate
        return {
            "src_ids": enc_src.ids[:self.max_len],
            "tgt_ids": enc_tgt.ids[:self.max_len]
        }

def get_or_build_tokenizer(config, sentences_list):
    """如果存在则加载，不存在则根据数据训练 Tokenizer"""
    if os.path.exists(config.tokenizer_path):
        print(f"Loading tokenizer from {config.tokenizer_path}...")
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() # 简单预分词，中文可能需要更复杂的
        
        trainer = BpeTrainer(
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
            vocab_size=config.vocab_size, 
            min_frequency=2
        )
        
        # 训练数据迭代器
        tokenizer.train_from_iterator(sentences_list, trainer=trainer)
        
        # 后处理：自动加 [SOS] 和 [EOS]
        tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )
        tokenizer.save(config.tokenizer_path)
        print("Tokenizer saved.")
    
    return tokenizer

def collate_fn(batch, pad_id):
    """处理 Batch 中的 Padding"""
    src_ids = [torch.tensor(item['src_ids']) for item in batch]
    tgt_ids = [torch.tensor(item['tgt_ids']) for item in batch]
    
    # 动态 Padding
    src_ids = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_ids = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    
    return src_ids, tgt_ids