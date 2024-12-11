from src.utils import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        assert self.config.d_embd % self.config.n_head == 0

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(self.config.d_embd, 3 * self.config.d_embd) # 3 for k, q, v

        # output projection
        self.c_proj = nn.Linear(self.config.d_embd, self.config.d_embd)

        self.c_proj.INIT_NORM = 1

    def forward(self, x):

        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (d_embd)

        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.config.d_embd, dim =2)

        # for MultiHead Attention, we split C(embedding) dimension to (n_head * hs)
        # nh (n_head) - number of heads, hs  - head size = C / n_head
        key = key.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        query = query.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        value = value.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Scaled dot-product Attention:
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # We are using Flash Attention here, which is more Optimised computation of Attention
        # Flash Attention & Attention are same, but pytorch executes flash attention efficiently

        # Flash Attention
        y = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B, T, C) # reassemble all head outputs into original shape

        # output projection
        y = self.c_proj(y)

        return y

class MultiLayerPerceptron(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.c_fc = nn.Linear(self.config.d_embd, 4 * self.config.d_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * self.config.d_embd, self.config.d_embd)

        self.c_proj.INIT_NORM = 1

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
    
class decoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.RMSnorm1 = nn.RMSNorm(self.config.d_embd)
        self.att = MultiHeadAttention(config)
        self.RMSnorm2 = nn.RMSNorm(self.config.d_embd)
        self.MLP = MultiLayerPerceptron(config)

    def forward(self, x, start_pos = None):

        # forward for RMS Norm 1 and Multi Head Attention
        x = x + self.att(self.RMSnorm1(x))

        # forward for RMS Norm 2 and Multi Layer Perceptron
        x = x + self.MLP(self.RMSnorm2(x))

        return x
    
# Model Architecture
class lyricGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(self.config.vocab_size, self.config.d_embd)
        self.wpe = nn.Embedding(self.config.n_seq, self.config.d_embd)

        self.decoder = nn.ModuleList([ decoderBlock(config) for _ in range(self.config.n_layer) ])

        self.RMSnorm = nn.RMSNorm(self.config.d_embd)

        self.ln_head = nn.Linear(self.config.d_embd, self.config.vocab_size, bias=False)

        # weight sharing
        self.wte.weight = self.ln_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'INIT_NORM'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):

        #input ids are of shape (B,T)
        B, T = idx.size()
        assert T<= self.config.n_seq, f"Input Sequence length({T}) is more than Maximum Sequence Length{self.config.n_seq}"

        #forward for Input and Positional Embeddings
        pos = torch.arange(0, T, dtype= torch.long, device=idx.device) # shape (T)
        pos_emb = self.wpe(pos) # Positional embedding of shape (T, d_embd)
        tok_emb = self.wte(idx) # Input embedding of shape (B, T, d_embd)
        x = tok_emb + pos_emb

        # forward for the Decoder Block of Transformer
        for decoder in self.decoder:
            x = decoder(x)

        # forward for RMS Norm before Classifier
        x = self.RMSnorm(x)

        # forward for Linear Classifier
        logits = self.ln_head(x) # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

if __name__ == '__main__':
    model = lyricGPT(Config())
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")