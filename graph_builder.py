import torch
from torch_geometric.data import Data

def graph_builder(flow_queue, graph_queue):

    while True:

        item = flow_queue.get()

        if item is None:
            graph_queue.put(None)
            break

        window, group = item

        group["y"] = group["Label"].apply(
            lambda x: 1 if "Botnet" in str(x) else 0
        )

        ips = set(group["SrcAddr"]) | set(group["DstAddr"])

        ip_to_id = {
            ip: idx
            for idx, ip in enumerate(ips)
        }

        src = group["SrcAddr"].map(ip_to_id).tolist()
        dst = group["DstAddr"].map(ip_to_id).tolist()

        edge_index = torch.tensor(
            [src, dst],
            dtype=torch.long
        )

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

        y = torch.tensor(
            group["y"].tolist(),
            dtype=torch.long
        )

        x = torch.ones(
            (len(ips), 4),
            dtype=torch.float
        )
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        graph_queue.put(graph)

        print(
            f"[Graph Update] Nodes={graph.num_nodes} "
            f"Edges={graph.num_edges}"
        )