import torch
import torch.nn as nn
import torch.nn.functional as F

class BioFormerPrompt(nn.Module):
    def __init__(self, vocab_size, num_cell_types, num_bins, d_model=512, nhead=8, num_layers=12, dropout=0.1, num_perturbations=500):
        super().__init__()
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.celltype_embedding = nn.Embedding(num_cell_types, d_model)
        self.perturbation_embedding = nn.Embedding(num_perturbations, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.cont_head = nn.Linear(d_model, 1)

    def forward(self, gene_idx, value_idx, celltype_idx, perturb_idx, attention_mask=None):
        gene_emb = self.gene_embedding(gene_idx)
        value_emb = self.value_embedding(value_idx)
        celltype_emb = self.celltype_embedding(celltype_idx)

        x = gene_emb + value_emb + celltype_emb

        # Add perturbation embedding as a [PERT] token prepended to sequence
        perturb_emb = self.perturbation_embedding(perturb_idx).unsqueeze(1)  # shape: (B, 1, D)
        x = torch.cat([perturb_emb, x], dim=1)

        if attention_mask is not None:
            pad = torch.ones((x.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)

        # Shift output index: CLS is now at position 1
        cls_rep = x[:, 1, :]

        return self.mlm_head(x), self.cont_head(x), x
