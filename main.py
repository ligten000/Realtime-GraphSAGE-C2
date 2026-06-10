import threading
import queue

from flow_builder import flow_builder
from graph_builder import graph_builder
from inference import inference

flow_queue = queue.Queue()
graph_queue = queue.Queue()

t1 = threading.Thread(
    target=flow_builder,
    args=(flow_queue,)
)

t2 = threading.Thread(
    target=graph_builder,
    args=(flow_queue, graph_queue)
)

t3 = threading.Thread(
    target=inference,
    args=(graph_queue,)
)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

print("System stopped.")