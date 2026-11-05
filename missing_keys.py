"""Print the comma-joined keys of a plan (optionally filtered by prefix) whose eval_alloc output is missing."""
import json
import os
import sys

plan, model, prefix, start = sys.argv[1:5]
keys = [k for k in json.load(open(plan))["allocations"] if k.startswith(start)]
d = f"metrics/alloc_eval/{model}_int8ent"
print(",".join(k for k in keys if not os.path.exists(os.path.join(d, prefix + k.replace("@", "_B") + ".json"))))
