import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEForEdgeClassification(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGEForEdgeClassification, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training) 
        h = F.relu(h)

        row, col = edge_index
        edge_feat_source = h[row]
        edge_feat_destination = h[col]

        edge_representation = torch.cat([edge_feat_source, edge_feat_destination], dim=-1)

        out = self.classifier(edge_representation).squeeze(-1)
        return out