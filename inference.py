import gc
import torch
from graph_model import GraphSAGEForEdgeClassification
from src.gnn_inference import execute_layer2_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphSAGEForEdgeClassification(in_channels=4, hidden_channels=64).to(device)
# Nạp file trọng số chất lượng cao vừa train từ Colab về
model.load_state_dict(torch.load("graphsage_best_model.pth", map_location=device))
model.eval()
print("[Model] GraphSAGE Real-time đạt chuẩn đã sẵn sàng.")

def inference(graph_queue):
    while True:
        graph = graph_queue.get()

        if graph is None:
            break

        graph = graph.to(device)

        with torch.no_grad():
            # 🎯 CHỈNH SỬA: Gọi model với ma trận graph.x thực tế đã chuẩn hóa từ graph_builder
            out = model(graph.x.float(), graph.edge_index)
            
            probs_tensor = torch.sigmoid(out)
            probs = probs_tensor.cpu().numpy().flatten()
            
            # In thống kê phân hóa xác suất thời gian thực
            if len(probs) > 0:
                print(f"📊 [AI Realtime] Score Cao nhất: {probs.max():.4f} | Thấp nhất: {probs.min():.4f} | Trung bình: {probs.mean():.4f}")

            preds = (probs > 0.5)
            botnet_edges = int(preds.sum())

            print(f"[Layer 1 AI] Edges={graph.num_edges} | Phát hiện nghi vấn thực tế={botnet_edges}")

            # Đóng gói dữ liệu đồ thị cửa sổ hiện tại để đẩy sang Layer 2
            window_id = getattr(graph, "window_id", "unknown_window")
            ips_list = getattr(graph, "ips_list", [])

            window_graph_data = {
                "window_id": window_id,
                "edges": []
            }

            edge_index_np = graph.edge_index.cpu().numpy()
            edge_attr_np = graph.edge_attr.cpu().numpy()

            for idx in range(graph.num_edges):
                src_idx = int(edge_index_np[0][idx])
                dst_idx = int(edge_index_np[1][idx])
                
                src_ip = ips_list[src_idx] if src_idx < len(ips_list) else "Unknown"
                dst_ip = ips_list[dst_idx] if dst_idx < len(ips_list) else "Unknown"
                features = edge_attr_np[idx]

                window_graph_data["edges"].append({
                    "src": src_ip,
                    "dst": dst_ip,
                    "features": features
                })

            # 🚀 LAYER 2: Đẩy sang bộ lọc luật thông minh để triệt tiêu 86% lượng bắt nhầm của AI
            execute_layer2_filter(window_graph_data, probs, output_json="alerts.json")
            print(f"[Layer 2 Filter] Đã hoàn tất rà soát bảo vệ cho cửa sổ {window_id}.\n")

        del graph
        gc.collect()