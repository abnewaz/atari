"""
Decision Transformer for Atari (image-based observations).

Architecture:
  1. CNN image encoder  → per-timestep state embedding
  2. Linear embedding   → return-to-go embedding
  3. Linear embedding   → action embedding
  4. Positional (timestep) embedding added to all three
  5. Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
  6. GPT-style causal transformer
  7. Predict action at each position from the state token output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CNN Encoder for Atari frames
# ─────────────────────────────────────────────
class AtariCNNEncoder(nn.Module):
    """
    Encode stacked Atari frames (C, 84, 84) into a flat feature vector.
    """

    def __init__(self, in_channels: int = 4, cnn_channels=(32, 64, 64),
                 cnn_kernels=(8, 4, 3), cnn_strides=(4, 2, 1),
                 output_dim: int = 128):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s))
            layers.append(nn.ReLU())
            ch_in = ch_out
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute the CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            cnn_flat_size = self.cnn(dummy).shape[1]

        self.fc = nn.Linear(cnn_flat_size, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W)  or  (batch, K, C, H, W)
        Returns:
            features: (..., output_dim)
        """
        # Handle sequence dimension
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.reshape(B * K, C, H, W)
            features = self.fc(self.cnn(x))
            features = features.reshape(B, K, self.output_dim)
        else:
            features = self.fc(self.cnn(x))
        return features


# ─────────────────────────────────────────────
# Causal Self-Attention
# ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        causal = self.causal_mask[:, :, :T, :T]
        attn = attn.masked_fill(causal == 0, float("-inf"))

        # Apply padding attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) → (B, 1, 1, T)
            attn = attn.masked_fill(
                attention_mask[:, None, None, :] == 0, float("-inf")
            )

        # Safe softmax: replace all-(-inf) rows with 0 to avoid NaN
        attn_max = attn.max(dim=-1, keepdim=True).values
        all_masked = (attn_max == float("-inf"))
        # Set all-masked rows to 0 before softmax so they produce uniform dist
        attn = attn.masked_fill(all_masked, 0.0)
        attn = F.softmax(attn, dim=-1)
        # Zero out the attention weights for fully masked rows (no info to attend to)
        attn = attn.masked_fill(all_masked, 0.0)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj_drop(self.proj(out))
        return out


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────
# Decision Transformer
# ─────────────────────────────────────────────
class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Atari.

    Input sequence (for context length K):
        [R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_K, s_K, a_K]

    Total sequence length = 3 * K tokens.

    We predict actions from the state token positions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.context_length = config.context_length
        self.n_actions = 4  # Breakout: NOOP, FIRE, RIGHT, LEFT

        max_seq_len = 3 * config.context_length  # R, s, a interleaved

        # ── Token Embeddings ──
        # State encoder (CNN)
        self.state_encoder = AtariCNNEncoder(
            in_channels=config.frame_stack,
            cnn_channels=config.cnn_channels,
            cnn_kernels=config.cnn_kernels,
            cnn_strides=config.cnn_strides,
            output_dim=config.embed_dim,
        )

        # Return-to-go embedding
        self.return_embed = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.Tanh(),
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Embedding(self.n_actions, config.embed_dim),
        )

        # ── Positional / Timestep Embeddings ──
        # Global timestep embedding (shared across R, s, a at same timestep)
        self.timestep_embed = nn.Embedding(config.max_ep_len, config.embed_dim)

        # Token-type embedding to distinguish R, s, a within a timestep
        # 0 = return, 1 = state, 2 = action
        self.token_type_embed = nn.Embedding(3, config.embed_dim)

        # ── Transformer ──
        self.embed_ln = nn.LayerNorm(config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.n_heads, config.dropout, max_seq_len
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embed_dim)

        # ── Prediction Head ──
        # Predict action from state token position
        self.action_head = nn.Linear(config.embed_dim, self.n_actions, bias=False)

        self.apply(self._init_weights)
        print(f"Decision Transformer — params: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        Args:
            states:        (B, K, C, H, W)
            actions:       (B, K)           int64
            returns_to_go: (B, K)           float
            timesteps:     (B, K)           int64
            attention_mask:(B, K)           float  (1=valid, 0=pad)

        Returns:
            action_logits: (B, K, n_actions)
        """
        B, K = states.shape[0], states.shape[1]

        # ── Compute token embeddings ──
        state_emb = self.state_encoder(states)                       # (B, K, D)
        action_emb = self.action_embed(actions)                      # (B, K, D)
        return_emb = self.return_embed(returns_to_go.unsqueeze(-1))  # (B, K, D)

        # ── Add timestep embeddings ──
        time_emb = self.timestep_embed(timesteps)  # (B, K, D)
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        return_emb = return_emb + time_emb

        # ── Add token-type embeddings ──
        type_ids = torch.arange(3, device=states.device)  # [0, 1, 2]
        type_emb = self.token_type_embed(type_ids)         # (3, D)

        return_emb = return_emb + type_emb[0]
        state_emb = state_emb + type_emb[1]
        action_emb = action_emb + type_emb[2]

        # ── Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...] ──
        # Shape each: (B, K, D) → stack to (B, K, 3, D) → reshape (B, 3K, D)
        token_stack = torch.stack([return_emb, state_emb, action_emb], dim=2)
        tokens = token_stack.reshape(B, 3 * K, self.embed_dim)

        # ── Build attention mask for interleaved sequence ──
        if attention_mask is not None:
            # Repeat each mask entry 3 times (for R, s, a)
            attn_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, 3 * K)
        else:
            attn_mask = None

        # ── Transformer forward ──
        tokens = self.drop(self.embed_ln(tokens))

        for block in self.blocks:
            tokens = block(tokens, attn_mask)

        tokens = self.ln_f(tokens)

        # ── Extract state token outputs ──
        # State tokens are at positions 1, 4, 7, ... (index 3*t + 1)
        state_token_indices = torch.arange(K, device=states.device) * 3 + 1  # (K,)
        state_tokens = tokens[:, state_token_indices, :]  # (B, K, D)

        # ── Predict actions ──
        action_logits = self.action_head(state_tokens)  # (B, K, n_actions)

        return action_logits

    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Inference helper: given a context, predict the next action.

        All inputs should be (1, T, ...) where T <= K.
        Returns: action (int)
        """
        K = self.context_length

        # Pad / truncate to context_length
        T = states.shape[1]
        if T > K:
            # Take last K steps
            states = states[:, -K:]
            actions = actions[:, -K:]
            returns_to_go = returns_to_go[:, -K:]
            timesteps = timesteps[:, -K:]
            T = K

        # Pad from the left if T < K
        if T < K:
            pad = K - T
            device = states.device

            states = torch.cat([
                torch.zeros(1, pad, *states.shape[2:], device=device),
                states
            ], dim=1)
            actions = torch.cat([
                torch.zeros(1, pad, dtype=torch.long, device=device),
                actions
            ], dim=1)
            returns_to_go = torch.cat([
                torch.zeros(1, pad, device=device),
                returns_to_go
            ], dim=1)
            timesteps = torch.cat([
                torch.zeros(1, pad, dtype=torch.long, device=device),
                timesteps
            ], dim=1)

            attention_mask = torch.cat([
                torch.zeros(1, pad, device=device),
                torch.ones(1, T, device=device)
            ], dim=1)
        else:
            attention_mask = torch.ones(1, K, device=states.device)

        action_logits = self.forward(states, actions, returns_to_go,
                                      timesteps, attention_mask)

        # The action prediction at the last valid position
        # (last position in the non-padded region)
        last_idx = K - 1  # since we padded from left, last position is always K-1
        logits = action_logits[0, last_idx]  # (n_actions,)
        action = torch.argmax(logits).item()
        return action