import torch
import torch_geometric
from torch_geometric.data import Data
import csv
import pandas as pd

print('Torch version:', torch.__version__)
print('Torch geometric version:', torch_geometric.__version__)
  
df = pd.read_csv('nodal_displacements.csv')
node_features = df[['X', 'Y', 'LOAD']]


x = torch.tensor(node_features.to_numpy(), dtype=torch.float)

edges = pd.read_csv('edges.csv')
edge_index = torch.tensor(edges.to_numpy(), dtype=torch.long)
edge_index = edge_index.t().contiguous()

print(x)
print(edge_index)


