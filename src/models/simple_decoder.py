"""
SimpleBiLSTMDecoder: A fallback decoder implementation for image captioning.

This decoder provides a minimal but functional implementation that can be used
when the full project decoder is unavailable or during development/testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBiLSTMDecoder(nn.Module):
    """
    Simple BiLSTM-based decoder for image captioning with basic attention.
    
    This is a fallback implementation that provides:
    - Word embeddings
    - LSTM-based sequential decoding
    - Basic attention over visual features
    - Teacher forcing support during training
    - Greedy generation for inference
    
    Attributes:
        embed_dim: Dimension of word embeddings (required by CaptioningModel)
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, 
                 visual_dim=768, num_layers=1, dropout=0.1, pad_idx=0, **kwargs):
        """
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for LSTM
            visual_dim: Dimension of visual features from encoder
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Padding token index
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.visual_dim = visual_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Project visual features to match hidden dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Simple attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # LSTM decoder
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
            visual_features: (B, N, visual_dim) - visual feature sequence
            captions: (B, max_len) - target captions (with <sos> at start)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: (B, max_len-1, vocab_size) - predicted logits
        """
        batch_size = visual_features.size(0)
        max_len = captions.size(1)
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)  # (B, N, hidden_dim)
        
        # Initialize LSTM hidden states
        device = visual_features.device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Start with <sos> token
        input_token = captions[:, 0]  # (B,)
        
        outputs = []
        
        for t in range(1, max_len):
            # Embed current input
            embedded = self.embedding(input_token)  # (B, embed_dim)
            embedded = self.dropout(embedded).unsqueeze(1)  # (B, 1, embed_dim)
            
            # Simple attention: use top layer hidden state
            hidden_state = h[-1].unsqueeze(1).expand(-1, visual_proj.size(1), -1)  # (B, N, hidden_dim)
            attention_input = torch.cat([visual_proj, hidden_state], dim=2)  # (B, N, 2*hidden_dim)
            attention_scores = self.attention(attention_input).squeeze(2)  # (B, N)
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, N)
            
            # Compute context vector
            context = torch.bmm(attention_weights.unsqueeze(1), visual_proj)  # (B, 1, hidden_dim)
            
            # Concatenate embedding and context
            lstm_input = torch.cat([embedded, context], dim=2)  # (B, 1, embed_dim + hidden_dim)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))  # (B, 1, hidden_dim)
            
            # Predict next word
            output = self.fc_out(lstm_out.squeeze(1))  # (B, vocab_size)
            outputs.append(output)
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = captions[:, t]
            else:
                input_token = output.argmax(dim=1)
        
        outputs = torch.stack(outputs, dim=1)  # (B, max_len-1, vocab_size)
        return outputs
    
    def generate(self, visual_features, max_len=50, sos_idx=1, eos_idx=2, beam_size=1):
        """
        Generate captions using greedy decoding.
        
        Args:
            visual_features: (B, N, visual_dim) - visual feature sequence
            max_len: Maximum caption length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            beam_size: Beam width (currently only supports greedy with beam_size=1)
        
        Returns:
            captions: (B, max_len) - generated caption token indices
        """
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
            hidden_state = h[-1].unsqueeze(1).expand(-1, visual_proj.size(1), -1)
            attention_input = torch.cat([visual_proj, hidden_state], dim=2)
            attention_scores = self.attention(attention_input).squeeze(2)
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), visual_proj)
            
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
