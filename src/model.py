import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.generator = nn.Linear(d_model, vocab_size)
        
        # Tied Embeddings
        self.src_embed.weight = self.tgt_embed.weight
        self.generator.weight = self.tgt_embed.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        
        # Transformer
        # src_key_padding_mask corresponds to Encoder Padding
        # memory_key_padding_mask corresponds to Encoder Padding during Cross-Attention
        outs = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask, 
            memory_mask=None, # 通常不需要 memory_mask
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.generator(outs)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        return self.transformer.encoder(
            self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model)), 
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return self.transformer.decoder(
            self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model)), 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )