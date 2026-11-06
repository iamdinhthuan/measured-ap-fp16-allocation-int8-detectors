# Build Supplementary table of the post-freeze proxy arms (A: per-image SQNR, C: block Fisher, FIT) from bootstrap JSONs.
import json, sys
M=["yolo11s","yolo26s","yolo11l","yolo26l"]; NAME={"yolo11s":"YOLO11-s","yolo26s":"YOLO26-s","yolo11l":"YOLO11-l","yolo26l":"YOLO26-l"}
B=["0.03","0.05","0.1","0.15","0.2"]
def p2(d,n):
    p=d["AP"]["p_le0"]; return max(2*min(p,1-p),2/(n+1))
def holm(ps):
    idx=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); adj=[0]*m; run=0
    for r,i in enumerate(idx):
        run=max(run,min(1,(m-r)*ps[i])); adj[i]=run
    return adj
rows={}; tests={}
for m in M:
    a=json.load(open(f"metrics/bootstrap/fx_{m}_a11.json")); c=json.load(open(f"metrics/bootstrap/fx_{m}_c11.json"))
    for b in B:
        r={}
        r["mean"]=a["point"][f"mean_B{b}"]["AP"]
        cm=c["point"][f"mean_B{b}"]["AP"]; assert abs(cm-r["mean"])<1e-9,(m,b)
        r["pool"]=a["point"][f"sqnrfx_B{b}"]["AP"]; r["pi"]=a["point"][f"pandey_B{b}"]["AP"]
        r["fit"]=c["point"][f"fit_B{b}"]["AP"]
        r["fblk"]=c["point"].get(f"fisherblk_B{b}",{}).get("AP")
        rows[(m,b)]=r
        for key,js,pair in [("pi",a,f"pandey_B{b}:mean_B{b}"),("fit",c,f"fit_B{b}:mean_B{b}")]:
            d=js["diff_b_minus_a"][pair]; lo,hi=d["AP"]["ci95"]
            tests.setdefault(key,[]).append(((m,b),lo>0 or hi<0, p2(d,js["n"]), hi<0))
mark={}
for key,L in tests.items():
    adj=holm([t[2] for t in L])
    for t,q in zip(L,adj):
        mark[(key,)+t[0]]=("" if not t[1] else ("^{\\dagger}" if t[3] else ("^{**}" if q<0.05 else "^{*}")))
    print(key,"sig",sum(t[1] for t in L),"holm",sum(q<0.05 and t[1] for t,q in zip(L,adj)),"proxy better",sum(t[3] for t in L),file=sys.stderr)
f=lambda x:"--" if x is None else f"{x:.2f}"
out=[]
for m in M:
    for j,b in enumerate(B):
        r=rows[(m,b)]; lab=f"\\multirow{{5}}{{*}}{{{NAME[m]}}}" if j==0 else ""
        out.append(f"{lab} & {int(round(float(b)*100))} & {f(r['mean'])} & {f(r['pool'])} & ${f(r['pi'])}{mark[('pi',m,b)]}$ & {f(r['fblk'])} & ${f(r['fit'])}{mark[('fit',m,b)]}$\\\\")
    out.append("\\midrule" if m!=M[-1] else "")
print("\n".join(out))
