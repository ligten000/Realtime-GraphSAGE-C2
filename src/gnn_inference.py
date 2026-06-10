import json
import os
import ipaddress
from datetime import datetime

class RuleEngineFilter:
    def __init__(self, config_path=None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "rules_config.json"))
        else:
            self.config_path = config_path
        self.load_rules()

    def load_rules(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
        else:
            # Luật dự phòng linh hoạt mặc định
            self.rules = {
                "trusted_subnets": ["127.0.0.0/8"],
                "min_packets_threshold": 2,
                "min_bytes_threshold": 500,
                "threat_intel_threshold": 70
            }

    def is_in_trusted_subnet(self, ip_str):
        """Kiểm tra một IP bất kỳ có thuộc dải mạng tin cậy hay không"""
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            for subnet_str in self.rules.get("trusted_subnets", []):
                if ip_obj in ipaddress.ip_network(subnet_str, strict=False):
                    return True
            return False
        except ValueError:
            return False # IP không hợp lệ

    def query_threat_intelligence(self, ip_str):
        """
        Mô phỏng kết nối API với VirusTotal / AlienVault.
        Thực tế sẽ trả về số nguồn bảo mật kết luận IP này là độc hại (0-100%).
        """
        # Giả lập: Nếu IP thuộc dải tấn công phổ biến trong dữ liệu mẫu CTU-13, trả về điểm độc hại cao
        if ip_str.startswith("147.32."):
            return 95 # 95% các chợ mã độc cảnh báo IP này
        return 10 # IP sạch hoặc chưa có vết

    def verify_edge(self, src_ip, dst_ip, features):
        """
        BỘ LỌC TINH THỰC TẾ KHÔNG CẦN WHITE-LIST CỨNG
        """
        # LUẬT 1: Lọc nhiễu mạng dựa trên số lượng gói tin và dung lượng tối thiểu
        tot_pkts = features[1]
        tot_bytes = features[2]
        if tot_pkts < self.rules["min_packets_threshold"] or tot_bytes < self.rules["min_bytes_threshold"]:
            return False, f"Báo động giả bị loại: Lưu lượng quá nhỏ ({tot_pkts} pkts, {tot_bytes} bytes)."

        # LUẬT 2: Nếu IP nằm trong dải mạng nội bộ tin cậy (Trusted Subnet)
        if self.is_in_trusted_subnet(src_ip):
            # Kiểm tra thêm Threat Intel để đề phòng trường hợp máy nội bộ bị chiếm quyền (Insider Threat)
            threat_score = self.query_threat_intelligence(dst_ip)
            if threat_score < self.rules["threat_intel_threshold"]:
                return False, f"Bỏ qua: IP nguồn thuộc Subnet nội bộ sạch và IP đích có điểm đe dọa thấp ({threat_score}%)."
            else:
                return True, f"🚨 NGUY HIỂM: Máy nội bộ đang kết nối đến C2 Server bên ngoài (Threat Intel: {threat_score}%)."

        # LUẬT 3: Đối với các IP lạ từ ngoài vào, kiểm tra danh tiếng qua Threat Intelligence
        threat_score = self.query_threat_intelligence(src_ip)
        if threat_score >= self.rules["threat_intel_threshold"]:
            return True, f"Xác thực hành vi: AI báo động + Threat Intelligence đồng thuận ({threat_score}% độc hại)."
        
        return False, f"Báo động giả bị loại: IP chưa có dấu hiệu độc hại trên hệ thống toàn cầu ({threat_score}%)."


def execute_layer2_filter(window_graph_data, ai_scores, output_json="alerts.json"):
    filter_engine = RuleEngineFilter()
    final_alerts = []
    edges_list = window_graph_data.get("edges", [])
    window_id = window_graph_data.get("window_id", "unknown_window")

    for idx, edge in enumerate(edges_list):
        ai_score = ai_scores[idx]

        # LAYER 1: Vượt qua bộ lọc AI GraphSAGE (Recall cao)
        if ai_score >= 0.85:
            src_ip = edge["src"]
            dst_ip = edge["dst"]
            features = edge["features"]

            # LAYER 2: Bộ lọc luật thông minh không White-list cứng
            is_valid_alert, detailed_reason = filter_engine.verify_edge(src_ip, dst_ip, features)

            if is_valid_alert:
                alert_log = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "window_id": window_id,
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                    "layer1_ai_score": float(round(ai_score, 4)),
                    "layer2_rule_passed": True,
                    "alert_level": "CRITICAL",
                    "reason": f"Phát hiện Botnet mức độ cao. {detailed_reason}"
                }
                final_alerts.append(alert_log)

    # Ghi file alerts.json
    if final_alerts:
        existing_data = []
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        existing_data.extend(final_alerts)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)