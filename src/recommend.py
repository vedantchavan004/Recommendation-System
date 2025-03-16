import torch
from torch_sparse import SparseTensor
from data import load_ratings
from model import LightGCN
from train import build_norm_adj, edge_index, num_users, num_items  # reuse variables from train.py

# Load ratings to rebuild norm_adj (or save norm_adj to file in a more advanced implementation)
ratings, num_users, num_items = load_ratings()
norm_adj = build_norm_adj(num_users, num_items, edge_index)

# Create model instance and load trained parameters
embedding_size = 64
num_layers = 2
model = LightGCN(num_users, num_items, embedding_size, num_layers)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def recommend(model, norm_adj, user_id, top_k=10):
    with torch.no_grad():
        user_emb, item_emb = model(norm_adj)
        scores = (user_emb[user_id].unsqueeze(0) * item_emb).sum(dim=1)
        _, top_items = torch.topk(scores, top_k)
    return top_items

# Example: Recommend for user 0
user_id = 0
recommended_items = recommend(model, norm_adj, user_id, top_k=10)
print(f"Top 10 recommendations for user {user_id}:", recommended_items.tolist())
