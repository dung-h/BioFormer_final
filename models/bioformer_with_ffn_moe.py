
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_layer_with_moe import TransformerEncoderLayerWithMoE

class BioFormerMoE(nn.Module):
    def __init__(self, vocab_size, num_cell_types, num_studies, num_bins=51,
                 d_model=512, nhead=8, num_layers=12, dropout=0.1,
                 num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins

        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)

        encoder_layer = TransformerEncoderLayerWithMoE(
            d_model=d_model, nhead=nhead, dropout=dropout, num_experts=num_experts
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Linear(d_model, num_bins)
        self.cont_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gene_embedding, self.value_embedding, self.cell_type_embedding]:
            nn.init.xavier_uniform_(emb.weight)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.xavier_uniform_(self.cont_head.weight)

    def forward(self, binned_expr, cell_type, study_id=None, non_zero_mask=None, return_attention=False):
        batch_size, seq_len = binned_expr.size()
        device = binned_expr.device

        gene_emb = self.gene_embedding(torch.arange(seq_len, device=device).expand(batch_size, seq_len))
        value_emb = self.value_embedding(binned_expr)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)

        emb = gene_emb + value_emb + cell_type_emb
        emb = self.norm(emb)

        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        route_weights_list = []

        # Pass through transformer layers manually to collect route weights
        for layer in self.transformer.layers:
            emb, route_weights = layer(emb)
            route_weights_list.append(route_weights)

        output = emb
        mlm_logits = self.mlm_head(output)
        cont_pred = self.cont_head(output).squeeze(-1)

        # Average route weights across all layers
        avg_route_weights = torch.stack(route_weights_list).mean(dim=0)  # (B, E)

        return mlm_logits, cont_pred, output, avg_route_weights
