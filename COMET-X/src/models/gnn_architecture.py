# src/models/gnn_architecture.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero, Linear # Example layers, adjust as per your model
from src.config import GNN_INPUT_DIM_CONTENT, GNN_INPUT_DIM_USER # Make sure these are imported

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Define different layers for each node type's initial projection
        self.lin_content = Linear(GNN_INPUT_DIM_CONTENT, hidden_channels) # Use the calculated input dim
        self.lin_user = Linear(GNN_INPUT_DIM_USER, hidden_channels) # Use the user input dim

        self.conv1 = SAGEConv((-1, -1), hidden_channels) # -1 means infer input dimension
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # This is a placeholder; actual forward pass logic will be in HeteroGNN
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Initial linear transformations for each node type
        # This maps the raw features to the same 'hidden_channels' dimension
        # before passing to HeteroConv.
        self.lin_dict = torch.nn.ModuleDict({
            'content': Linear(GNN_INPUT_DIM_CONTENT, hidden_channels), # Ensure this matches
            'user': Linear(GNN_INPUT_DIM_USER, hidden_channels),     # Ensure this matches
        })

        # Define the heterogeneous graph convolutional layers
        # Use 'SAGEConv' for the actual message passing.
        # The -1 means `SAGEConv` will infer the input dimension from the first message passing step.
        self.convs = torch_geometric.nn.to_hetero(SAGEConv((-1, -1), hidden_channels), metadata=[
            ('user', 'posts', 'content')
            # Add other relations if you define them later, e.g., ('content', 'reposts', 'content')
        ])
        # A final linear layer for classification after graph convolutions
        # This layer takes the output features of the 'content' nodes (which will be 'hidden_channels' if you use a 1-layer conv or 'out_channels' if multiple convs output to out_channels)
        # and maps them to the final classification dimension (e.g., 2 for fake/real).
        self.lin = Linear(hidden_channels, out_channels) # Assuming convs output to hidden_channels

    def forward(self, x_dict, edge_index_dict):
        # Apply initial linear transformations to all node types
        x_dict_transformed = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }
        # Pass through the heterogeneous graph convolution layers
        # Make sure the output of convs matches the input to the final linear layer
        # In your case, `self.convs` should output `hidden_channels` features for content nodes
        x_dict_conv = self.convs(x_dict_transformed, edge_index_dict)

        # Get the output for 'content' nodes and pass through the final linear layer
        # Assuming 'content' is the node type you're classifying
        out = self.lin(x_dict_conv['content'])
        return out