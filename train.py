import torch
import torch.nn as nn
import numpy as np
from graph_model import GraphSAGEForEdgeClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Trình tính toán đang sử dụng GPU: {device}")

# 1. Nạp tập dữ liệu thô ban đầu từ Drive
dataset = torch.load("all_graphs.pt", weights_only=False)

# Phân chia tập dữ liệu huấn luyện
train_size = min(60, len(dataset))
train_graphs = dataset[:train_size]
test_graphs = dataset[train_size:]
# 2. Khởi tạo mô hình
model = GraphSAGEForEdgeClassification(in_channels=4, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 🎯 SỬA LỖI GỐC: Giảm phạt nhãn sai xuống 10.0 để tránh sinh trọng số âm cực đoan
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))

print("🚀 Bắt đầu quá trình huấn luyện mô hình mới...")
model.train()

for epoch in range(1, 26): # Chạy 25 epoch trên GPU cực nhanh
    total_loss = 0
    for graph in train_graphs:
        graph = graph.to(device)

        # 🎯 THUẬT TOÁN ĐỒNG BỘ: Tính toán ma trận nút x động bằng cơ chế Log-Transform
        num_nodes = graph.num_nodes
        node_features_sum = torch.zeros((num_nodes, 4), dtype=torch.float, device=device)
        node_degree = torch.zeros((num_nodes, 1), dtype=torch.float, device=device)

        # Áp dụng log1p trực tiếp vào đặc trưng cạnh để nén thang đo byte/packet
        log_edge_attr = torch.log1p(graph.edge_attr.float())
        
        row, col = graph.edge_index
        node_features_sum.index_add_(0, row, log_edge_attr)

        ones = torch.ones((graph.edge_index.shape[1], 1), dtype=torch.float, device=device)
        node_degree.index_add_(0, row, ones)
        node_degree = torch.clamp(node_degree, min=1.0)

        # Trích xuất ma trận nút x có cùng phân phối nhỏ gọn
        x_nodes = node_features_sum / node_degree

        optimizer.zero_grad()
        out = model(x_nodes, graph.edge_index)

        loss = criterion(out, graph.y.float())
        loss.backward()
        
        # Clip gradient tránh bùng nổ đạo hàm gây lỗi NaN/Inf
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d}/25 | Loss trung bình: {total_loss / len(train_graphs):.4f}")
torch.save(model.state_dict(), "graphsage_best_model.pth")
