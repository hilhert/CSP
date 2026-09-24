import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Shared embedding and head
# ============================================================
class BaseSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.requires_grad = False 
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.head       = nn.Linear(hidden_dim, output_dim)
        self._init_embedding()

    def _init_embedding(self):
        nn.init.orthogonal_(self.embed.weight,gain=1.0)
    
    def _embed(self, x):
        # x: [B, T, 1] or [B, T]
        x = x.squeeze(-1).long()               # [B, T]
        emb = self.embed(x)                    # [B, T, embed_dim]
        return self.embed_proj(emb)            # [B, T, hidden_dim]

    def _head(self, h):
        return self.head(self.dropout(h))      # [B, T, output_dim]


# ============================================================
# 1. Vanilla RNN
# ============================================================
class VanillaRNN(BaseSeqModel):
    def __init__(self, vocab_size, embed_dim=16,hidden_dim=32, output_dim=2, dropout=0.1):
        super().__init__(vocab_size, embed_dim,hidden_dim, output_dim, dropout)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True,
                          nonlinearity='tanh')

    def forward(self, x):
        # x: [B, T, 1] long or float; cast to long for embedding
        x = x.squeeze(-1).long()
        emb = self._embed(x)              # [B, T, H]
        h, _ = self.rnn(emb)             # [B, T, H]
        return self._head(h)


# ============================================================
# 2. Complex RNN
#    State is complex-valued, updated as
#      h_t = h_{t-1} * e^{i theta_t} + W z_t
#    Implemented with real/imag channels.
# ============================================================
class ComplexRNN(BaseSeqModel):
    """
    Complex-valued RNN with per-channel rotation and nonlinear state update.

    Update:
        theta_t = pi * tanh(W_theta z_t + b_theta)          # [B, H]
        h_t     = h_{t-1} * e^{i theta_t} + W_in z_t       # complex
        h_t     = h_t + g_t * tanh(phi(h_t))                # nonlinearity
    Readout:
        y_t = head([Re(h_t), Im(h_t)])
    """
    def __init__(self, vocab_size, embed_dim=16 ,hidden_dim=32, output_dim=2, dropout=0.1):
        super().__init__(vocab_size, embed_dim,hidden_dim, output_dim, dropout)
        H = hidden_dim
        self.H = H

        # Per-channel rotation angle (scalar per channel, not one per vector)
        self.theta_proj = nn.Linear(H, H)

        # Complex input projection: (re, im) from real embedding
        self.in_proj = nn.Linear(H, 2 * H)

        # Nonlinearity applied to the complex state, per-channel gate
        self.state_gate = nn.Linear(2 * H, H)        # gate in (0,1)
        self.state_mix  = nn.Linear(2 * H, 2 * H)    # nonlinear candidate

        # LayerNorm over real+imag state
        self.state_norm = nn.LayerNorm(2 * H)

        # Readout
        self.head = nn.Linear(2 * H, output_dim)

    def forward(self, x):
        x = x.squeeze(-1).long()
        emb = self._embed(x)                          # [B, T, H]
        B, T, H = emb.shape
        dev = x.device

        h_re = torch.zeros(B, H, device=dev)
        h_im = torch.zeros(B, H, device=dev)

        outs = []
        for t in range(T):
            z = emb[:, t, :]                          # [B, H]

            # --- per-channel rotation ---
            theta = torch.pi * torch.tanh(self.theta_proj(z))   # [B, H]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            h_re_rot = cos_t * h_re - sin_t * h_im
            h_im_rot = sin_t * h_re + cos_t * h_im

            # --- complex input injection ---
            inp = self.in_proj(z)                     # [B, 2H]
            inp_re, inp_im = inp[:, :H], inp[:, H:]

            # pre-activation complex state
            h_re_pre = h_re_rot + inp_re
            h_im_pre = h_im_rot + inp_im

            # --- nonlinear state update (this is what was missing) ---
            cat = torch.cat([h_re_pre, h_im_pre], dim=-1)   # [B, 2H]
            gate = torch.sigmoid(self.state_gate(cat))      # [B, H]
            cand = torch.tanh(self.state_mix(cat))          # [B, 2H]
            cand_re, cand_im = cand[:, :H], cand[:, H:]
            h_re = h_re_pre + gate * cand_re
            h_im = h_im_pre + gate * cand_im

            # --- normalize to keep |h| bounded ---
            h_cat = torch.cat([h_re, h_im], dim=-1)
            h_cat = self.state_norm(h_cat)
            h_re, h_im = h_cat[:, :H], h_cat[:, H:]

            outs.append(h_cat)

        h = torch.stack(outs, dim=1)                  # [B, T, 2H]
        return self.head(self.dropout(h))


# ============================================================
# 3. Transformer (decoder-only, causal)
# ============================================================
class TransformerModel(BaseSeqModel):
    def __init__(self, vocab_size,embed_dim=16, hidden_dim=32, output_dim=2,
                 n_heads=4, n_layers=2, dropout=0.1, max_len=128):
        super().__init__(vocab_size, embed_dim , hidden_dim, output_dim, dropout)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=2*hidden_dim,
            dropout=dropout, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.max_len = max_len

    def forward(self, x):
        x = x.squeeze(-1).long()
        B, T = x.shape
        emb = self._embed(x)
        pos = self.pos(torch.arange(T, device=x.device)).unsqueeze(0)
        emb = emb + pos
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # no memory: pass emb as memory and ignore cross-attention
        h = self.decoder(emb, emb, tgt_mask=mask)
        return self._head(h)


# ============================================================
# 4. Mamba (simplified)
#    Selective SSM with input-dependent A, B, C.
#    A is diagonal and negative to keep the recurrence stable.
# ============================================================
class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, d_state=16, d_conv=3, expand=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_inner = int(expand * hidden_dim)

        self.in_proj = nn.Linear(hidden_dim, 2 * self.d_inner)
        self.conv1d = CausalConv1d(self.d_inner, d_conv)
        # input-dependent dt, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(d_state * 2 + 1, self.d_inner)
        # A is diagonal, parameterized in log space
        self.A_raw = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, hidden_dim)

    def forward(self, x):
        # x: [B, T, H]
        B, T, H = x.shape
        xz = self.in_proj(x)                      # [B, T, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)
        # depthwise conv
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # input-dependent parameters
        x_dbl = self.x_proj(x_conv)               # [B, T, 2*d_state+1]
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(x_dbl))      # [B, T, d_inner]

        A = torch.tanh(self.A_raw)                # [d_inner, d_state]

        # discretize
        dA = torch.tanh(dt.unsqueeze(-1) * A)      # [B, T, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, T, d_inner, d_state]
        dBx = dB * x_conv.unsqueeze(-1)

        # scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)                # [B, T, d_inner]
        y = y + self.D * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaModel(BaseSeqModel):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32, output_dim=2,
                 n_layers=3, dropout=0.1):
        super().__init__(vocab_size, embed_dim,hidden_dim, output_dim, dropout)
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = x.squeeze(-1).long()
        emb = self._embed(x)                       # [B, T, H]
        h = emb
        for layer in self.layers:
            h = h + layer(h)                      # residual
        return self._head(h)