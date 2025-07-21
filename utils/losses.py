import torch
import torch.nn.functional as F

 
def bio_consistency_loss(embeddings, cell_type):
    
    embeddings = embeddings[:, 0]  # Use CLS token
    dist_matrix = torch.cdist(embeddings, embeddings)
    same_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    loss = (same_type * dist_matrix.pow(2)).mean()
    return loss

def contrastive_loss(embeddings, cell_type, study_id, temperature=0.5):
    
    embeddings = embeddings[:, 0]  # Use CLS token
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    exp_sim = torch.exp(sim_matrix)
    
    same_cell_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    different_study = (study_id.unsqueeze(1) != study_id.unsqueeze(0)).float()
    pos_mask = same_cell_type * different_study
    
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    total_sim = exp_sim.sum(dim=1)
    loss = -torch.log(pos_sim / total_sim + 1e-8).mean()
    return loss

def ecs_loss(embeddings, cell_type, margin=1.0):
    """Enhanced Cell Similarity (ECS) loss for cell type consistency."""
    # Use CLS token embeddings
    embeddings = embeddings[:, 0] if embeddings.dim() > 2 else embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings)
    
    # Create masks for same and different cell types
    same_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    diff_type = 1.0 - same_type
    
    # Remove diagonal (self-similarity)
    mask = 1.0 - torch.eye(embeddings.size(0), device=embeddings.device)
    same_type = same_type * mask
    diff_type = diff_type * mask
    
    # Positive pairs (same cell type) should be close
    pos_loss = (same_type * dist_matrix.pow(2)).sum() / (same_type.sum() + 1e-8)
    
    # Negative pairs (different cell type) should be far apart  
    neg_loss = (diff_type * F.relu(margin - dist_matrix).pow(2)).sum() / (diff_type.sum() + 1e-8)
    
    return pos_loss + neg_loss