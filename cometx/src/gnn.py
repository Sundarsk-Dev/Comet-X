import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

class GNN(torch.nn.Module):
    """
    A simple two-layer heterogeneous GNN using SAGEConv.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def build_hetero_graph(num_posts, num_users, post_features, user_features, user_posts_edges):
    """
    Builds a simple heterogeneous graph with 'post' and 'user' nodes.
    """
    data = HeteroData()
    data['post'].x = post_features
    data['user'].x = user_features
    data['user', 'posts', 'post'].edge_index = user_posts_edges
    data['post', 'posted_by', 'user'].edge_index = user_posts_edges.flip([0])
    return data