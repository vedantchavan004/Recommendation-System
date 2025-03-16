import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size, num_layers):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, norm_adj):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        
        for _ in range(self.num_layers):
            all_embeddings = norm_adj.matmul(all_embeddings)
            embeddings_list.append(all_embeddings)
        
        final_embedding = sum(embeddings_list) / (self.num_layers + 1)
        user_emb, item_emb = final_embedding.split([self.num_users, self.num_items])
        return user_emb, item_emb
    
    def get_score(self, user_emb, item_emb, users, items):
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        return (u_emb * i_emb).sum(dim=1)
