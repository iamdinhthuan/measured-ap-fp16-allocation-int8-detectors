"""rules_11_12d_add1.md queues. rd_queue.py unchanged; its module globals are rebound per stage.
Usage: python rd_queueX.py STAGE   (bs: A3 proxy, base int8ent | F: A2 measured arm, base int8entF)"""
import os
import sys

import rd_queue as q

stage = sys.argv[1]
if stage == "bs":
    for B in q.BUD:
        q.task(os.path.join("configs", f"fx_{q.M}_{q.BASE}_ho_noise_n2000bs.json"), f"sqnrq@{B}", "fxbs_")
elif stage == "F":
    q.BASE = "int8entF"
    q.D = os.path.join("metrics", "alloc_eval", f"{q.M}_{q.BASE}")
    for B in q.BUD:
        q.task(os.path.join("metrics", f"alloc_{q.M}_{q.BASE}_ho.json"), f"mean@{B}")
else:
    raise SystemExit(stage)
print("QUEUE_DONE", stage, flush=True)
