import torch
from torch_geometric.data import Data

def graph_builder(flow_queue, graph_queue):

    while True:
        item = flow_queue.get()

        if item is None:
            graph_queue.put(None)
            break

        window, group = item

        # Tạo nhãn số hóa: 1 cho Botnet, 0 cho Luồng sạch
        group["y"] = group["Label"].apply(
            lambda x: 1 if "Botnet" in str(x) else 0
        )

        # Định nghĩa tập các IP (Nodes) xuất hiện trong cửa sổ dữ liệu
        ips = set(group["SrcAddr"]) | set(group["DstAddr"])

        ip_to_id = {
            ip: idx
            for idx, ip in enumerate(ips)
        }

        src = group["SrcAddr"].map(ip_to_id).tolist()
        dst = group["DstAddr"].map(ip_to_id).tolist()

        # Ma trận chỉ mục cạnh (Edge Index)
        edge_index = torch.tensor(
            [src, dst],
            dtype=torch.long
        )

        # Trích xuất đặc trưng cạnh thô ban đầu
        edge_attr = torch.tensor(
            group[
                [
                    "Dur",
                    "TotPkts",
                    "TotBytes",
                    "SrcBytes"
                ]
            ].values,
            dtype=torch.float
        )

        # 🎯 ĐỊNH NGHĨA BIẾN Y CHUẨN XÁC ĐỂ FIX LỖI PYLANCE
        y = torch.tensor(
            group["y"].tolist(),
            dtype=torch.long
        )

        # --- KHỐI TOÁN HỌC LOG-TRANSFORM ĐỒNG BỘ VỚI COLAB ---
        num_nodes = len(ips)
        node_features_sum = torch.zeros((num_nodes, 4), dtype=torch.float)
        node_degree = torch.zeros((num_nodes, 1), dtype=torch.float)

        # Áp dụng hàm nén phân phối log1p đồng bộ tuyệt đối với mô hình AI
        log_edge_attr = torch.log1p(edge_attr.float())
        node_features_sum.index_add_(0, edge_index[0], log_edge_attr)

        ones = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
        node_degree.index_add_(0, edge_index[0], ones)
        node_degree = torch.clamp(node_degree, min=1.0)

        # Ma trận đặc trưng nút x đại diện cấu trúc phân phối nhỏ gọn
        x = node_features_sum / node_degree
        # -----------------------------------------------------

        # Ánh xạ chỉ mục quay ngược lại chuỗi IP để phục vụ Layer 2 hiển thị alert
        ips_list = [None] * len(ips)
        for ip, idx in ip_to_id.items():
            ips_list[idx] = ip

        # Đóng gói đồ thị hình học PyG
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,                 # 🟢 Biến y đã được truyền vào hợp lệ
            window_id=str(window),
            ips_list=ips_list   
        )

        graph_queue.put(graph)
        print(f"[Graph Builder] Tạo đồ thị Window {window} thành công (Nodes={graph.num_nodes}, Edges={graph.num_edges})")
# import torch
# from torch_geometric.data import Data

# def graph_builder(flow_queue, graph_queue):

#     while True:

#         item = flow_queue.get()

#         if item is None:
#             graph_queue.put(None)
#             break

#         window, group = item

#         group["y"] = group["Label"].apply(
#             lambda x: 1 if "Botnet" in str(x) else 0
#         )

#         ips = set(group["SrcAddr"]) | set(group["DstAddr"])

#         ip_to_id = {
#             ip: idx
#             for idx, ip in enumerate(ips)
#         }

#         src = group["SrcAddr"].map(ip_to_id).tolist()
#         dst = group["DstAddr"].map(ip_to_id).tolist()

#         edge_index = torch.tensor(
#             [src, dst],
#             dtype=torch.long
#         )

#         edge_attr = torch.tensor(
#             group[
#                 [
#                     "Dur",
#                     "TotPkts",
#                     "TotBytes",
#                     "SrcBytes"
#                 ]
#             ].values,
#             dtype=torch.float
#         )

#         num_nodes = len(ips)
#         node_features_sum = torch.zeros((num_nodes, 4), dtype=torch.float)
#         node_degree = torch.zeros((num_nodes, 1), dtype=torch.float)

#         # 🎯 CHỈNH SỬA: Áp dụng log1p trực tiếp để nén dải phân phối dữ liệu
#         log_edge_attr = torch.log1p(edge_attr.float())
#         node_features_sum.index_add_(0, edge_index[0], log_edge_attr)

#         ones = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
#         node_degree.index_add_(0, edge_index[0], ones)
#         node_degree = torch.clamp(node_degree, min=1.0)

#         # Ma trận thuộc tính nút x đã được chuẩn hóa đồng bộ
#         x = node_features_sum / node_degree

#         ips_list = [None] * len(ips)
#         for ip, idx in ip_to_id.items():
#             ips_list[idx] = ip

#         graph = Data(
#             x=x,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=y,
#             window_id=str(window),
#             ips_list=ips_list   
#         )

#         graph_queue.put(graph)
#         print(f"[Graph Builder] Tạo đồ thị Window {window} thành công (Nodes={graph.num_nodes}, Edges={graph.num_edges})")