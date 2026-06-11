import subprocess
import pandas as pd
import time


def flow_builder_live(flow_queue):

    print("[Live Mode] Starting tcpdump...")

    process = subprocess.Popen(
        ["sudo", "tcpdump", "-i", "any", "-nn"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    flows = []

    start_window = pd.Timestamp.now()

    for line in process.stdout:

        if "IP" not in line:
            continue

        try:

            parts = line.split()

            src = parts[2]
            dst = parts[4].replace(":", "")

            src_ip = src.rsplit(".", 1)[0]
            dst_ip = dst.rsplit(".", 1)[0]

            flow = {
                "SrcAddr": src_ip,
                "DstAddr": dst_ip,
                "Dur": 1,
                "TotPkts": 1,
                "TotBytes": 64,
                "SrcBytes": 64,
                "Label": "Normal"
            }

            flows.append(flow)

        except Exception:
            continue

        now = pd.Timestamp.now()

        if (now - start_window).seconds >= 30:

            group = pd.DataFrame(flows)

            flow_queue.put(
                (start_window.floor("5min"), group)
            )

            print(
                f"[Live Flow Builder] "
                f"Window={start_window.floor('5min')} "
                f"Flows={len(group)}"
            )

            flows = []
            start_window = now

    flow_queue.put(None)