from gnn_inference import execute_layer2_filter

# 1. Tự tạo dữ liệu giả lập giống hệt như dữ liệu dòng chảy mạng thật để kiểm thử
mock_window_data = {
    "window_id": "window_demo_001",
    "edges": [
        {"src": "8.8.8.8", "dst": "192.168.1.5", "features": [0.01, 10, 5000, 2500]},
        
        {"src": "147.32.84.165", "dst": "147.32.84.192", "features": [0.5, 1, 64, 64]},
        
        # Trường hợp 3: AI nghi ngờ cực cao, thỏa mãn tất cả luật -> Sẽ được giữ lại
        {"src": "147.32.84.200", "dst": "147.32.84.165", "features": [12.4, 45, 15200, 8400]}
    ]
}

# Giả lập điểm số xác suất từ mô hình GraphSAGE trả về cho 3 luồng mạng trên
mock_ai_scores = [0.91, 0.88, 0.94]

# 2. Chạy thử nghiệm hàm lọc luật
print("=== BẮT ĐẦU KIỂM THỬ MODULE LỌC LUẬT ===")
execute_layer2_filter(mock_window_data, mock_ai_scores)
print("=== KIỂM THỬ HOÀN TẤT, HÃY KIỂM TRA FILE ALERTS.JSON ===")