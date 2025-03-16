import random
import torch
import torch.optim as optim
from torch_sparse import SparseTensor
import pandas as pd
from data import download_and_extract, load_ratings
from model import LightGCN

# Download and load data
download_and_extract()
ratings, num_users, num_items = load_ratings()

# Build edge_index tensor
edge_index = torch.tensor(ratings[['user', 'item']].values.T, dtype=torch.long)

def build_norm_adj(num_users, num_items, edge_index):
    edge_index = edge_index.clone()
    edge_index[1] += num_users  # shift item indices
    
    row, col = edge_index
    rev_edge_index = torch.stack([col, row], dim=0)
    full_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    
    N = num_users + num_items
    adj = SparseTensor(row=full_edge_index[0], col=full_edge_index[1], sparse_sizes=(N, N))
    
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    row, col, val = adj.coo()
    if val is None:
        val = torch.ones_like(row, dtype=torch.float)
    
    norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
    norm_adj = SparseTensor(row=row, col=col, value=norm_val, sparse_sizes=(N, N))
    
    return norm_adj

norm_adj = build_norm_adj(num_users, num_items, edge_index)

# Create LightGCN model
embedding_size = 64
num_layers = 2
model = LightGCN(num_users, num_items, embedding_size, num_layers)

# Prepare training: BPR loss and negative sampling
user_item_dict = ratings.groupby('user')['item'].apply(set).to_dict()

def sample_negative(user):
    while True:
        neg_item = random.randint(0, num_items - 1)
        if neg_item not in user_item_dict[user]:
            return neg_item

def bpr_loss(user_emb, item_emb, batch):
    users, pos_items, neg_items = [], [], []
    for (user, pos_item) in batch:
        users.append(user)
        pos_items.append(pos_item)
        neg_items.append(sample_negative(user))
        
    users = torch.tensor(users, dtype=torch.long)
    pos_items = torch.tensor(pos_items, dtype=torch.long)
    neg_items = torch.tensor(neg_items, dtype=torch.long)
    
    pos_scores = model.get_score(user_emb, item_emb, users, pos_items)
    neg_scores = model.get_score(user_emb, item_emb, users, neg_items)
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
pos_pairs = list(ratings[['user', 'item']].itertuples(index=False, name=None))
num_epochs = 50
batch_size = 1024

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    user_emb, item_emb = model(norm_adj)
    batch = random.sample(pos_pairs, min(batch_size, len(pos_pairs)))
    loss = bpr_loss(user_emb, item_emb, batch)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

print("Training finished!")

torch.save(model.state_dict(), "model.pth")
