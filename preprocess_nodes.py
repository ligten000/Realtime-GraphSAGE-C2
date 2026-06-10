import torch

dataset = torch.load("all_graphs.pt", weights_only=False)

for graph in dataset:
    num_nodes = graph.num_nodes
    num_edge_features = graph.edge_attr.shape[1] # Bằng 4

    # Tạo ma trận chứa tổng giá trị và ma trận đếm số lượng kết nối của mỗi Node
    node_features_sum = torch.zeros((num_nodes, num_edge_features), dtype=torch.float)
    node_degree = torch.zeros((num_nodes, 1), dtype=torch.float)

    row, col = graph.edge_index

    # Bước a: Cộng dồn giá trị vào Node nguồn
    node_features_sum.index_add_(0, row, graph.edge_attr.float())

    # Bước b: Đếm xem Node nguồn đó xuất hiện bao nhiêu lần (bậc của Node)
    ones = torch.ones((graph.edge_index.shape[1], 1), dtype=torch.float)
    node_degree.index_add_(0, row, ones)

    # Bước c: Tránh chia cho 0 (máy không có kết nối nào)
    node_degree = torch.clamp(node_degree, min=1.0)

    # Bước d: TÍNH TRUNG BÌNH CỘNG (Sự khác biệt cốt lõi)
    node_features_mean = node_features_sum / node_degree

    # Gán lại vào thuộc tính x của đồ thị
    graph.x = node_features_mean

# 2. Lưu thành file mới
torch.save(dataset, "all_graphs_fixed_node.pt")