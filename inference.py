import gc
import torch

from graph_model import GraphSAGEForEdgeClassification

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = GraphSAGEForEdgeClassification(
    in_channels=4,
    hidden_channels=64
).to(device)

model.load_state_dict(
    torch.load(
        "graphsage_best_model.pth",
        map_location=device
    )
)

model.eval()

print("[Model] GraphSAGE loaded")


def inference(graph_queue):

    while True:

        graph = graph_queue.get()

        if graph is None:
            break

        graph = graph.to(device)

        with torch.no_grad():

            out = model(
                graph.x.float(),
                graph.edge_index
            )
            
            probs = torch.sigmoid(out)

            preds = (probs > 0.5).long()

            botnet_edges = int(preds.sum())

            print(
                f"[Inference] "
                f"Edges={graph.num_edges} | "
                f"Botnet={botnet_edges}"
            )

        del graph
        gc.collect()

        print("[Memory] Graph released")