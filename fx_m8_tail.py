"""M8 helper: print the mean@B keys whose blocks differ from the published plan (comma-separated, may be empty)."""
import json
import sys

new = json.load(open(sys.argv[1]))["allocations"]
pub = json.load(open(sys.argv[2]))["allocations"]
diff = [k for k in new if k.startswith("mean@") and new[k]["blocks"] != pub[k]["blocks"]]
print(",".join(diff))
