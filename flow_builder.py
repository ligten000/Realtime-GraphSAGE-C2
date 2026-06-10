import pandas as pd
import time

def flow_builder(flow_queue):

    df = pd.read_csv("capture20110810.binetflow")

    df["StartTime"] = pd.to_datetime(df["StartTime"])
    df["window"] = df["StartTime"].dt.floor("5min")

    for window, group in df.groupby("window"):

        flow_queue.put((window, group))

        print(f"[Flow Builder] Window {window} pushed")

        time.sleep(1)

    flow_queue.put(None)