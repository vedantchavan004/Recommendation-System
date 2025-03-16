import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from src.model import GCN

# Load the dataset
dataset = Planetoid(root='../data/Cora', name='Cora')
data = dataset[0]

# Initialize the model and load the saved parameters
model = GCN(hidden_channels=16)
model.load_state_dict(torch.load('../data/best_model.pth'))
model.eval()

# Testing function
def test():
    out = model(data)
    pred = out.argmax(dim=1)
    test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    test()
