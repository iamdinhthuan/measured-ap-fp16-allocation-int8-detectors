"""Writes tables/headfirst.tex (Supplementary Table S8) and tables/fsweep.tex (Table S12) from the bootstrap files
metrics/bootstrap/fx_<model>_y1.json, fx_yolo26l_m8.json, m8l_yolo11l_fsweep.json and metrics/x1_exactcost.json.
Restored MAC shares of output SQNR, head-first SQNR and whole head are those of Table 2 of the article.
Usage: python supp_headfirst_fsweep_tables.py [out_dir]
"""
import json
import os,sys
ROOT=os.path.dirname(os.path.abspath(__file__))
R=os.path.join(ROOT,'metrics','bootstrap')+'/'
X=json.load(open(os.path.join(ROOT,'metrics','x1_exactcost.json')))['models']
OUT=(sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,'manuscript_inputs','tables'))+'/'
os.makedirs(OUT,exist_ok=True)
def p2(p): return max(2*min(p,1-p),2/1001)
def fmt(x):
    if abs(x)<0.005: return "$0.00$"
    return f"{x:+.2f}".replace('-','$-$') if False else (f"$+{x:.2f}$" if x>=0 else f"$-{-x:.2f}$")
def ci(c): 
    f=lambda v: (f"+{v:.2f}" if v>=0 else f"-{-v:.2f}")
    return f"$[{f(c[0])},{f(c[1])}]$"
def holm(ps,alpha=0.05):
    idx=sorted(range(len(ps)),key=lambda i:ps[i]); rej=[False]*len(ps); m=len(ps)
    for k,i in enumerate(idx):
        if ps[i]<=alpha/(m-k)+1e-12: rej[i]=True
        else: break
    return rej
mac_sqnr={'yolo11s':{.15:14.9,.2:19.8},'yolo26s':{.1:8.8,.15:13.6,.2:18.8},'yolo11l':{.1:9.5,.15:14.7,.2:19.7},'yolo26l':{.1:9.6,.15:14.3,.2:18.6}}
mac_hyb={'yolo11s':{.15:14.9,.2:19.3},'yolo26s':{.1:9.8,.15:14.6,.2:19.1},'yolo11l':{.1:10.0,.15:14.3,.2:19.5},'yolo26l':{.1:10.0,.15:14.2,.2:18.2}}
mac_head={'yolo11s':14.9,'yolo26s':8.9,'yolo11l':7.7,'yolo26l':7.6}
names={'yolo11s':'YOLO11-s','yolo26s':'YOLO26-s','yolo11l':'YOLO11-l','yolo26l':'YOLO26-l'}
rows={}
for m in ['yolo11l','yolo26l','yolo26s','yolo11s']:
    d=json.load(open(R+f'fx_{m}_y1.json')); pt=d['point']; df=d['diff_b_minus_a']
    for B in sorted(mac_hyb[m]):
        b=f"B{B:g}" if False else f"B{B}"
        k=lambda s: f"{s}_B{B}"
        A=lambda s: pt[k(s)]['AP']
        c1=df[f"{k('hyb')}:{k('mean')}"]['AP']; c2=df[f"{k('sqnrq')}:{k('hyb')}"]['AP']; c3=df[f"{k('head')}:{k('hyb')}"]['AP']
        mac_mean=100*X[m]['arms']['published'][f"mean@{B}"]['exact_mac_published']
        rows[(m,B)]=dict(mean=(A('mean'),mac_mean),hyb=(A('hyb'),mac_hyb[m][B]),head=(A('head'),mac_head[m]),sq=(A('sqnrq'),mac_sqnr[m][B]),
            d1=(A('mean')-A('hyb'),c1['ci95'],p2(c1['p_le0'])), d2=(A('hyb')-A('sqnrq'),c2['ci95'],p2(c2['p_le0'])), d3=(A('hyb')-A('head'),c3['ci95'],p2(c3['p_le0'])))
large=[r for r in rows if r[0] in('yolo11l','yolo26l')]
for key in ['d1','d2','d3']:
    rej=holm([rows[r][key][2] for r in large])
    for r,h in zip(large,rej): rows[r][key+'h']=h
def cell(r,key):
    v,c,p=rows[r][key]; s=fmt(v)
    sig = c[0]>0 or c[1]<0
    mk='' if not sig else ('^{**}' if rows[r].get(key+'h') else '^{*}')
    if mk: s=s[:-1]+mk+'$'
    return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{s}\\\\[-1pt]{{\\tiny{ci(c)}}}\\end{{tabular}}"
L=["\\begin{tabular}{@{}llllllll@{}}","\\toprule",
 "Model & $B$ & Measured & Head-first & Whole head & Raw SQNR & Meas.$-$HF & HF$-$raw & HF$-$head\\\\".replace('llllllll','lllllllll'),"\\midrule"]
L[0]="\\begin{tabular}{@{}llllllccc@{}}"
prev=None
for r in [('yolo11l',.1),('yolo11l',.15),('yolo11l',.2),('yolo26l',.1),('yolo26l',.15),('yolo26l',.2),('yolo26s',.1),('yolo26s',.15),('yolo26s',.2),('yolo11s',.15),('yolo11s',.2)]:
    m,B=r; x=rows[r]
    if prev and prev!=m: L.append("\\midrule")
    nm=names[m] if prev!=m else ''
    ap=lambda t: f"{t[0]:.2f} ({t[1]:.1f})"
    L.append(f"{nm} & {int(B*100)} & {ap(x['mean'])} & {ap(x['hyb'])} & {ap(x['head'])} & {ap(x['sq'])} & {cell(r,'d1')} & {cell(r,'d2')} & {cell(r,'d3')}\\\\")
    prev=m
L+=["\\bottomrule","\\end{tabular}"]
open(OUT+'headfirst.tex','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
# ---- F-sweep table
F=[]
specs=[('yolo26l',R+'fx_yolo26l_m8.json','mean_B{B}','fsw_B{B}','fsw_B{B}:mean_B{B}',-1),
       ('yolo11l',R+'m8l_yolo11l_fsweep.json','pub_{B}','fsw_{B}','pub_{B}:fsw_{B}',+1)]
L=["\\begin{tabular}{@{}lllllll@{}}","\\toprule",
   " & & \\multicolumn{2}{c}{AP (MAC, \\%)} & & \\multicolumn{2}{c}{Exact cost: selection changes?}\\\\","\\cmidrule(lr){3-4}\\cmidrule(l){6-7}",
   "Model & $B$ & Main & $\\Fset$ restored & $\\Fset$ restored$-$main & Main & $\\Fset$ restored\\\\","\\midrule"]
for i,(m,f,ka,kb,kd,sgn) in enumerate(specs):
    d=json.load(open(f)); pt=d['point']; df=d['diff_b_minus_a']
    ps=[];rs=[]
    for B in [0.03,0.05,0.1,0.15,0.2]:
        a=pt[ka.format(B=B)]['AP']; b=pt[kb.format(B=B)]['AP']; c=df[kd.format(B=B)]['AP']
        cc=c['ci95'] if sgn>0 else [-c['ci95'][1],-c['ci95'][0]]
        ps.append(p2(c['p_le0'])); rs.append((B,a,b,b-a,cc))
    rej=holm(ps)
    if i: L.append("\\midrule")
    for j,(B,a,b,dd,cc) in enumerate(rs):
        xm=X[m]['arms']['published'][f'mean@{B}']; xf=X[m]['arms']['fsweep'][f'mean@{B}']
        ch=lambda x: ((f"yes (${x['pred_dAP_tune']:+.2f}$)" if abs(x['pred_dAP_tune'])>=0.01 else f"yes (${x['pred_dAP_tune']:+.3f}$)") if x['changed'] else "no")
        s=fmt(dd); sig=cc[0]>0 or cc[1]<0
        if sig: s=s[:-1]+('^{**}' if rej[j] else '^{*}')+'$'
        nm=names[m] if j==0 else ''
        L.append(f"{nm} & {int(round(B*100))} & {a:.2f} ({100*xm['exact_mac_published']:.1f}) & {b:.2f} ({100*xf['exact_mac_published']:.1f}) & {s} {{\\tiny{ci(cc)}}} & {ch(xm)} & {ch(xf)}\\\\")
L+=["\\bottomrule","\\end{tabular}"]
open(OUT+'fsweep.tex','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
