import onnx, re, sys
from collections import Counter, defaultdict
m=onnx.load(sys.argv[1])
g=m.graph
print("opset",[(o.domain,o.version) for o in m.opset_import])
print("in",[(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("out",[(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.output])
print(Counter(n.op_type for n in g.node).most_common(30))
grp=defaultdict(Counter)
for n in g.node:
    mm=re.match(r"/model\.(\d+)/",n.name)
    k=mm.group(1) if mm else "other"
    grp[k][n.op_type]+=1
for k in sorted(grp,key=lambda x:(len(x),x)):
    c=grp[k]; print(k, sum(c.values()), dict(c.most_common(6)))
# decoder sub paths
sub=Counter()
for n in g.node:
    if n.name.startswith("/model.28/"):
        parts=n.name.split("/")[2:5]
        sub["/".join(p for p in parts[:2])]+=1
for k,v in sorted(sub.items()): print("  28:",k,v)
print([n.name for n in g.node if not n.name.startswith("/model.")][:20])
