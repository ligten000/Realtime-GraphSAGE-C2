import torch
import torch.nn as nn
from src.graph_model import GraphSAGEForEdgeClassification 
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Mô hình AI đang chạy trên: {device}")

train_size = 60
train_graphs = dataset[:train_size]
test_graphs = dataset[train_size:]

model = GraphSAGEForEdgeClassification(in_channels=4, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]).to(device))

model.train()
for epoch in range(1, 16):
    total_loss = 0
    for graph in train_graphs:
        graph = graph.to(device)

        if graph.x is None or graph.x.shape[0] == 0 or len(graph.x.shape) < 2 or graph.x.shape[1] != 4:
            x_nodes = torch.ones((graph.num_nodes, 4), dtype=torch.float, device=device)
        else:
            x_nodes = graph.x.float().to(device)

        x_nodes = x_nodes.view(graph.num_nodes, 4)

        optimizer.zero_grad()

        model = model.float()
        out = model(x_nodes, graph.edge_index)

        loss = criterion(out, graph.y.float())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d}/{15} | Loss trung bình: {total_loss / len(train_graphs):.4f}")
