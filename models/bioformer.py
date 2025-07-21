import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Token-Wise MoE Layer
# ----------------------
class TokenWiseMoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        logits = self.router(x)  # (B, L, E)
        route_weights = F.softmax(logits, dim=-1)  # (B, L, E)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # (B, L, D)
            output += expert_out * route_weights[..., i].unsqueeze(-1)

        return output, route_weights


# ----------------------
# Transformer Layer + MoE
# ----------------------
class TransformerEncoderLayerWithMoE(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.moe = TokenWiseMoE(d_model=d_model, num_experts=num_experts)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))

        ff_out, moe_weights = self.moe(src)
        src = self.norm2(src + self.dropout2(ff_out))

        return src, moe_weights


# ----------------------
# Full BioFormer Model
# ----------------------
class BioFormer(nn.Module):
    def __init__(self, vocab_size, num_cell_types, num_studies, num_bins=51,
                 d_model=512, nhead=8, num_layers=12, dropout=0.1, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.num_layers = num_layers

        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithMoE(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                num_experts=num_experts
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, num_bins)
        self.cont_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gene_embedding, self.value_embedding, self.cell_type_embedding]:
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.xavier_uniform_(self.cont_head.weight)

    def forward(self, binned_expr, cell_type, study_id=None, non_zero_mask=None, return_attention=False):
        B, L = binned_expr.shape
        device = binned_expr.device

        gene_emb = self.gene_embedding(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))  # (B, L, D)
        value_emb = self.value_embedding(binned_expr)  # (B, L, D)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)  # (B, 1, D)

        emb = gene_emb + value_emb + cell_type_emb
        emb = self.norm(emb)

        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        all_route_weights = []

        for layer in self.encoder_layers:
            emb, route_weights = layer(emb)
            all_route_weights.append(route_weights)  # each: (B, L, E)

        mlm_logits = self.mlm_head(emb)  # (B, L, num_bins)
        cont_pred = self.cont_head(emb).squeeze(-1)  # (B, L)

        # average routing over layers and tokens for logging
        if return_attention:
            avg_routing = torch.stack(all_route_weights, dim=0).mean(dim=(0, 1, 2))  # (E,)
            return mlm_logits, cont_pred, emb, avg_routing
        else:
            avg_routing = torch.stack(all_route_weights, dim=0).mean(dim=(0, 1, 2))  # (E,)
            return mlm_logits, cont_pred, emb, avg_routing
    
    