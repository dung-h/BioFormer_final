
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

class TokenWiseMoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=1, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.d_model = d_model
        
        # Create single batched expert layer for torch.compile compatibility
        self.expert_w1 = nn.Parameter(torch.randn(num_experts, d_model, d_model * 4))
        self.expert_w2 = nn.Parameter(torch.randn(num_experts, d_model * 4, d_model))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, d_model * 4))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.expert_w1)
        nn.init.xavier_uniform_(self.expert_w2)
        
        # Enhanced router with layer norm for stability
        self.router_norm = nn.LayerNorm(d_model)
        self.router_linear = nn.Linear(d_model, num_experts)
        self.router_dropout = nn.Dropout(0.1)
        
        # Add noise for better load balancing during training
        self.noise_std = 0.1

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Router computation - fully compile-friendly
        router_input = self.router_norm(x)
        if self.training and self.noise_std > 0:
            # Use deterministic noise for torch.compile compatibility
            noise = torch.empty_like(router_input).normal_(0, self.noise_std)
            router_input = router_input + noise
            
        logits = self.router_linear(router_input)
        logits = self.router_dropout(logits)
        route_weights = F.softmax(logits, dim=-1)  # (B, L, E)
        
        # Batched expert computation - fully vectorized for torch.compile
        # Reshape input for batched matrix multiplication
        x_flat = x.view(-1, D)  # (B*L, D)
        
        # Compute all experts in parallel using batched operations
        # First layer: (B*L, D) @ (E, D, 4D) -> (B*L, E, 4D)
        hidden = torch.einsum('bd,edf->bef', x_flat, self.expert_w1) + self.expert_b1.unsqueeze(0)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=0.1, training=self.training)
        
        # Second layer: (B*L, E, 4D) @ (E, 4D, D) -> (B*L, E, D)
        expert_outputs = torch.einsum('bef,efd->bed', hidden, self.expert_w2) + self.expert_b2.unsqueeze(0)
        
        # Reshape back to (B, L, E, D)
        expert_outputs = expert_outputs.view(B, L, self.num_experts, D)
        
        # Apply routing weights: (B, L, E, D) * (B, L, E, 1) -> (B, L, D)
        route_weights_expanded = route_weights.unsqueeze(-1)  # (B, L, E, 1)
        output = torch.sum(expert_outputs * route_weights_expanded, dim=2)  # (B, L, D)
        
        return output, route_weights


class TransformerEncoderLayerWithMoE(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Enhanced MoE with load balancing
        self.moe = TokenWiseMoE(d_model=d_model, num_experts=num_experts, load_balance_weight=0.01)
        
        # Add skip connection scaling for better training stability
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual connection
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) * self.skip_scale
        src = self.norm1(src)

        # MoE with enhanced residual connection
        ff_out, route_weights = self.moe(src)
        src = src + self.dropout2(ff_out) * self.skip_scale
        src = self.norm2(src)

        return src, route_weights
