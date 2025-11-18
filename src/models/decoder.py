import torch
import torch.nn as nn

# Channel attention (SE-like)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, N, C)
        b, n, c = x.size()
        y = x.permute(0,2,1)  # (B, C, N)
        y = self.avgpool(y).squeeze(-1)  # (B, C)
        y = self.fc(y).unsqueeze(1)  # (B,1,C)
        return x * y  # broadcast over N

# MDSA-C: Multi-Head Self-Attention + Channel attention gating
class MDSA_C(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_att = ChannelAttention(embed_dim, reduction=16)

    def forward(self, x):
        # x: (B, N, C)
        attn_out, _ = self.mha(x, x, x)  # self-attention across tokens
        x = self.norm1(x + attn_out)
        ff = self.ff(x)
        x = self.norm2(x + ff)
        # channel gating
        x = self.channel_att(x)
        return x

# BiLSTM encoder over tokens feeding into MDSA-C
class BiLSTM_MDSA_C_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1, num_heads=8):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        # BiLSTM outputs 2*hidden_dim, match attention embed dim
        self.mdsac = MDSA_C(embed_dim=2*hidden_dim, num_heads=num_heads)
    def forward(self, feat_seq):
        # feat_seq: (B, N, C)
        out, _ = self.bilstm(feat_seq)  # (B, N, 2*hidden_dim)
        out = self.mdsac(out)           # (B, N, 2*hidden_dim)
        return out

# Example integration with a decoder LSTM for word generation. This is an outline: your repo likely has its own decoder.
class SimpleCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + visual_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        # attention between decoder hidden state and visual tokens
        self.attn_proj = nn.Linear(visual_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, prev_word, hx, cx, visual_tokens):
        # prev_word: (B,) idx; visual_tokens: (B, N, C)
        emb = self.embed(prev_word)  # (B, E)
        # compute attention weights (dot-product)
        proj_visual = self.attn_proj(visual_tokens)  # (B,N,H)
        # hidden broadcast
        hidden = hx.unsqueeze(1)  # (B,1,H)
        attn_scores = (proj_visual * hidden).sum(dim=2)  # (B,N)
        attn_w = self.softmax(attn_scores)
        context = (attn_w.unsqueeze(2) * visual_tokens).sum(dim=1)  # (B,C)
        input_lstm = torch.cat([emb, context], dim=1)
        hx, cx = self.lstm(input_lstm, (hx, cx))
        out = self.fc(hx)
        return out, hx, cx, attn_w