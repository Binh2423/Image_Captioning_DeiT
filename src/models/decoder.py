import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel attention (SE-like)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, N, C)
        b, n, c = x.size()
        y = x.permute(0,2,1)  # (B, C, N)
        y = self.avgpool(y).squeeze(-1)  # (B, C)
        y = self.fc(y).unsqueeze(1)  # (B,1,C)
        return x * y  # broadcast over N

# MDSA-C: Multi-Head Dual-Stage Attention with Context
class MDSA_C(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Channel attention gating
        self.channel_att = ChannelAttention(embed_dim, reduction=16)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, N, C)
        # Stage 1: Multi-head self-attention
        attn_out, attn_weights = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Stage 2: Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Context gating with channel attention
        x = self.channel_att(x)
        
        return x, attn_weights

# BiLSTM encoder over tokens feeding into MDSA-C
class BiLSTM_MDSA_C_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=1, num_heads=8, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        # BiLSTM outputs 2*hidden_dim, match attention embed dim
        self.mdsac = MDSA_C(embed_dim=2*hidden_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, feat_seq, lengths=None):
        # feat_seq: (B, N, C)
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(feat_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.bilstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.bilstm(feat_seq)  # (B, N, 2*hidden_dim)
        
        # Create padding mask for attention if lengths provided
        key_padding_mask = None
        if lengths is not None:
            batch_size = feat_seq.size(0)
            max_len = feat_seq.size(1)
            key_padding_mask = torch.arange(max_len, device=feat_seq.device).unsqueeze(0).expand(batch_size, -1) >= lengths.unsqueeze(1)
        
        out, attn_weights = self.mdsac(out, key_padding_mask=key_padding_mask)  # (B, N, 2*hidden_dim)
        return out, attn_weights

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


# Complete BiLSTM Decoder with MDSA-C Attention and Teacher Forcing
class BiLSTMDecoder(nn.Module):
    """
    BiLSTM-based decoder with MDSA-C attention mechanism for image captioning.
    Supports teacher forcing during training and beam search during inference.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim, 
                 num_layers=1, dropout=0.1, num_heads=8, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.visual_dim = visual_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Attention mechanism for visual features
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Project visual features to match hidden dim
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # LSTM decoder (single direction for caption generation)
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
    
    def forward(self, visual_features, captions, teacher_forcing_ratio=1.0):
        """
        Forward pass with teacher forcing.
        
        Args:
            visual_features: (B, N, visual_dim) - visual feature sequence from encoder
            captions: (B, max_len) - target captions (with <sos> at start)
            teacher_forcing_ratio: probability of using teacher forcing (1.0 = always use)
        
        Returns:
            outputs: (B, max_len, vocab_size) - predicted logits for each position
        """
        batch_size = visual_features.size(0)
        max_len = captions.size(1)
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)  # (B, N, hidden_dim)
        
        # Initialize LSTM hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=visual_features.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=visual_features.device)
        
        # Initialize first input with <sos> token (first token in captions)
        input_token = captions[:, 0]  # (B,)
        
        outputs = []
        
        for t in range(1, max_len):
            # Embed current input
            embedded = self.embedding(input_token)  # (B, embed_dim)
            embedded = self.dropout(embedded).unsqueeze(1)  # (B, 1, embed_dim)
            
            # Compute attention over visual features
            # Query: top layer hidden state
            query = h[-1].unsqueeze(1)  # (B, 1, hidden_dim)
            context, attn_weights = self.visual_attention(
                query, visual_proj, visual_proj
            )  # context: (B, 1, hidden_dim)
            
            # Concatenate embedding and context
            lstm_input = torch.cat([embedded, context], dim=2)  # (B, 1, embed_dim + hidden_dim)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))  # lstm_out: (B, 1, hidden_dim)
            
            # Predict next word
            output = self.fc_out(lstm_out.squeeze(1))  # (B, vocab_size)
            outputs.append(output)
            
            # Teacher forcing: use ground truth with probability teacher_forcing_ratio
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = captions[:, t]  # Use ground truth
            else:
                input_token = output.argmax(dim=1)  # Use prediction
        
        outputs = torch.stack(outputs, dim=1)  # (B, max_len-1, vocab_size)
        
        return outputs
    
    def generate(self, visual_features, max_len=50, sos_idx=1, eos_idx=2, beam_size=1):
        """
        Generate captions using greedy decoding or beam search.
        
        Args:
            visual_features: (B, N, visual_dim) - visual feature sequence
            max_len: maximum caption length
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_size: beam width (1 = greedy decoding)
        
        Returns:
            captions: (B, max_len) - generated caption token indices
        """
        if beam_size == 1:
            return self._greedy_decode(visual_features, max_len, sos_idx, eos_idx)
        else:
            return self._beam_search(visual_features, max_len, sos_idx, eos_idx, beam_size)
    
    def _greedy_decode(self, visual_features, max_len, sos_idx, eos_idx):
        """Greedy decoding"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)
        
        # Initialize LSTM states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Start with <sos> token
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        
        captions = [input_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            # Embed current token
            embedded = self.embedding(input_token).unsqueeze(1)  # (B, 1, embed_dim)
            
            # Attention
            query = h[-1].unsqueeze(1)
            context, _ = self.visual_attention(query, visual_proj, visual_proj)
            
            # LSTM step
            lstm_input = torch.cat([embedded, context], dim=2)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Predict next token
            output = self.fc_out(lstm_out.squeeze(1))
            input_token = output.argmax(dim=1)
            
            # Mark sequences that produced <eos>
            finished |= (input_token == eos_idx)
            
            # Stop if all sequences finished
            if finished.all():
                break
            
            captions.append(input_token)
        
        captions = torch.stack(captions, dim=1)  # (B, T)
        return captions
    
    def _beam_search(self, visual_features, max_len, sos_idx, eos_idx, beam_size):
        """Beam search decoding (simplified for single batch)"""
        # For simplicity, implement beam search for batch_size=1
        assert visual_features.size(0) == 1, "Beam search currently only supports batch_size=1"
        
        device = visual_features.device
        visual_proj = self.visual_proj(visual_features)  # (1, N, hidden_dim)
        
        # Initialize
        h = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
        
        # Beam: list of (sequence, score, h, c)
        beams = [([sos_idx], 0.0, h, c)]
        completed = []
        
        for _ in range(max_len - 1):
            candidates = []
            
            for seq, score, h_state, c_state in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue
                
                # Get last token
                input_token = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                embedded = self.embedding(input_token).unsqueeze(1)
                
                # Attention
                query = h_state[-1].unsqueeze(1)
                context, _ = self.visual_attention(query, visual_proj, visual_proj)
                
                # LSTM step
                lstm_input = torch.cat([embedded, context], dim=2)
                lstm_out, (h_new, c_new) = self.lstm(lstm_input, (h_state, c_state))
                
                # Get top k predictions
                output = self.fc_out(lstm_out.squeeze(1))
                log_probs = F.log_softmax(output, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)
                
                for k in range(beam_size):
                    new_seq = seq + [topk_ids[0, k].item()]
                    new_score = score + topk_log_probs[0, k].item()
                    candidates.append((new_seq, new_score, h_new, c_new))
            
            # Select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # Stop if all beams completed
            if not beams:
                break
        
        # Add remaining beams to completed
        completed.extend(beams)
        
        # Return best sequence
        if completed:
            best_seq, _ = max(completed, key=lambda x: x[1] / len(x[0]))  # Normalize by length
            return torch.tensor([best_seq], dtype=torch.long, device=device)
        else:
            return torch.tensor([[sos_idx]], dtype=torch.long, device=device)