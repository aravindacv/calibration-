"""Post hoc analyses (paper Section 6.6): label-free gap estimates (ATC, corruption regression),
BBSD vs alarm on batches of 200, alarm+ATC rule, and label-budget curve.
Usage: python posthoc_analysis.py results     (folder with preds_*.npz and corr_cal_logits_*.npz)
Runs on CPU in a few minutes."""
import numpy as np
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize_scalar
from scipy.stats import wasserstein_distance
import sys, os, glob
# Locate the result files: a folder given on the command line, ./results, /kaggle/working,
# or any attached Kaggle result dataset (files may be spread over several datasets).
_ROOTS = [a for a in sys.argv[1:] if os.path.isdir(a)] + ['results', 'results/results', '/kaggle/working']
_ROOTS += [p for p in glob.glob('/kaggle/input/**/', recursive=False)] + glob.glob('/kaggle/input/datasets/*/*/')
_INDEX = {}
for _r in _ROOTS:
    if not os.path.isdir(_r) or any(k in _r.lower() for k in ('skinmilk', 'ham10000', 'pad-ufes', 'isic', 'milkyd')):
        continue
    for _dp, _, _fs in os.walk(_r):
        for _f in _fs:
            if _f.endswith(('.npz', '.csv', '.jsonl')):
                _INDEX.setdefault(_f, os.path.join(_dp, _f))
def F(name):
    if name not in _INDEX:
        raise FileNotFoundError(name + ' not found; attach the result datasets or pass the results folder')
    return _INDEX[name]
D = ''   # kept for compatibility; paths are resolved with F()
SETS=['test_id','shift_trunk','ext_ham','ext_pad']
KINDS=['clean','blur','jpeg','color','noise','lowres','combo']
CZ={**dict(np.load(F('corr_cal_logits_v2.npz'))),**dict(np.load(F('corr_cal_logits_fresh.npz')))}
def models():
    for bb,pre in [('B0',''),('B3','b3_')]:
        for v in ['ce','focal']:
            for s in range(6):
                yield bb,v,s,np.load(F(f'preds_{pre}{v}_s{s}.npz'))
def fitT(z,y): return minimize_scalar(lambda t:-np.mean(log_softmax(z/t,1)[np.arange(len(y)),y]),bounds=(0.05,10),method='bounded').x
def conf(z,T=1.0): return softmax(z/T,1).max(1)
def ece(z,y,T=1.0,nb=15):
    p=softmax(z/T,1);c=p.max(1);a=(p.argmax(1)==y);e=0;b=np.linspace(0,1,nb+1)
    for lo,hi in zip(b[:-1],b[1:]):
        m=(c>lo)&(c<=hi)
        if m.any(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def pick(zt,bb,v,s):
    ct=conf(zt); d={k:wasserstein_distance(ct,conf(CZ[f'{bb}_{v}_s{s}__{k}'])) for k in KINDS}
    return min(d,key=d.get), d


# ---- check: reimplemented matcher reproduces the frozen confirmatory picks
import pandas as pd
M=pd.read_csv(F('m3_rows.csv')); ok=tot=0
for bb,v,s,d in models():
    if s<3: continue
    for st in SETS:
        k,_=pick(d['z_'+st],bb,v,s); tot+=1; ok+=(k==M[(M.bb==bb)&(M.v==v)&(M.seed==s)&(M.set==st)].pick.iloc[0])
print('matcher agrees with frozen picks:',ok,'/',tot)

# ---- gap estimates and refit benefit
import pandas as pd
rng=np.random.default_rng(0); rows=[]
for bb,v,s,d in models():
    zc,yc=d['z_cal'],d['y_cal']; Ts=fitT(zc,yc)
    cc=conf(zc,Ts); acc_c=(zc.argmax(1)==yc).mean()
    t_atc=np.quantile(cc,1-acc_c)               # fraction above t equals cal accuracy
    # corruption-based regression: confidence drop -> gap, learned on labelled corrupted source
    X=[];G=[]
    for k in KINDS:
        zk=CZ[f'{bb}_{v}_s{s}__{k}']; ok=np.isfinite(zk).all(1); zk,yk=zk[ok],yc[ok]; ck=conf(zk,Ts)
        X.append(cc.mean()-ck.mean()); G.append(ck.mean()-(zk.argmax(1)==yk).mean())
    b1,b0=np.polyfit(X,G,1)
    for st in SETS:
        z,y=d['z_'+st],d['y_'+st]; ct=conf(z,Ts); acc=(z.argmax(1)==y).mean()
        e_clean=ece(z,y,Ts)
        # target-fit benefit (100 labels, 20 draws)
        ef=[]
        for r in range(20):
            p=rng.permutation(len(y)); f,e=p[:100],p[100:]
            ef.append(ece(z[e],y[e],fitT(z[f],y[f])))
        rows.append(dict(bb=bb,v=v,seed=s,set=st,acc=acc,cbar=ct.mean(),gap=ct.mean()-acc,ece_clean=e_clean,ece_tf=np.mean(ef),
            acc_doc=acc_c-(cc.mean()-ct.mean()), acc_atc=(ct>t_atc).mean(),
            gap_cr=b0+b1*(cc.mean()-ct.mean()), w1=wasserstein_distance(ct,cc)))
R=pd.DataFrame(rows)
R['gap_doc']=R.cbar-R.acc_doc; R['gap_atc']=R.cbar-R.acc_atc; R['benefit']=R.ece_clean-R.ece_tf
R.to_csv('a1_rows.csv',index=False)
pd.set_option('display.width',200)
g=R.groupby('set')[['acc','acc_doc','acc_atc','gap','gap_doc','gap_atc','gap_cr','ece_clean','ece_tf','benefit','w1']].mean().reindex(SETS)
print(g.round(3).to_string())
for est in ['gap_doc','gap_atc','gap_cr']:
    print(est,'MAE vs true gap by set:',R.assign(err=(R[est]-R.gap).abs()).groupby('set').err.mean().reindex(SETS).round(3).to_dict(),
          '| corr with ece_clean:',round(np.corrcoef(R[est],R.ece_clean)[0,1],3))


# ---- source-only threshold, BBSD, label budget
import pandas as pd
from scipy.stats import ks_2samp
rng=np.random.default_rng(1); rows=[]; bud=[]
for bb,v,s,d in models():
    zc,yc=d['z_cal'],d['y_cal']; Ts=fitT(zc,yc); cc=conf(zc,Ts); acc_c=(zc.argmax(1)==yc).mean()
    t_atc=np.quantile(cc,1-acc_c)
    # source-only null for the ATC gap: bootstrap the held-out in-domain set (labels not used)
    ci=conf(d['z_test_id'],Ts)
    null=[ (lambda c: c.mean()-(c>t_atc).mean())(rng.choice(ci,len(ci))) for _ in range(500)]
    tau=np.quantile(null,0.99)
    pc=softmax(zc,1)
    for st in SETS:
        z,y=d['z_'+st],d['y_'+st]; ct=conf(z,Ts); g=ct.mean()-(ct>t_atc).mean()
        k,_=pick(z,bb,v,s); alarm=k not in ('clean','color')
        # BBSD: KS on each softmax dim (T=1), Bonferroni, batches of 200 target images
        pt=softmax(z,1); det=[]
        for r in range(20):
            idx=rng.choice(len(y),200,replace=False)
            det.append(min(ks_2samp(pc[:,j],pt[idx,j]).pvalue for j in range(8))<0.05/8)
        rows.append(dict(bb=bb,v=v,seed=s,set=st,tau=tau,gap_atc=g,flag_atc=g>tau,alarm=alarm,flag_both=alarm and g>tau,bbsd=np.mean(det)))
        if st in ('ext_ham','ext_pad'):
            for m in [10,25,50,100,200]:
                for r in range(20):
                    p=rng.permutation(len(y)); f,e=p[:m],p[m:]
                    bud.append(dict(bb=bb,v=v,seed=s,set=st,m=m,ece=ece(z[e],y[e],fitT(z[f],y[f]))))
R=pd.DataFrame(rows); R.to_csv('a2_rows.csv',index=False); B=pd.DataFrame(bud); B.to_csv('a2_budget.csv',index=False)
print('tau (source-only 99th pct) min/median/max:',round(float(R.tau.min()),3),round(float(R.tau.median()),3),round(float(R.tau.max()),3))
print(R.groupby('set')[['alarm','flag_atc','flag_both']].sum().reindex(SETS))
print('BBSD detection rate (batches of 200):',R.groupby('set').bbsd.mean().reindex(SETS).round(2).to_dict())
print(B.groupby(['set','m']).ece.agg(['mean','std']).round(3).unstack(0))


# ---- alarm and alarm+ATC on batches of 200
import pandas as pd
rng=np.random.default_rng(2); rows=[]
for bb,v,s,d in models():
    zc,yc=d['z_cal'],d['y_cal']; Ts=fitT(zc,yc); cc=conf(zc,Ts); acc_c=(zc.argmax(1)==yc).mean(); t_atc=np.quantile(cc,1-acc_c)
    ci=conf(d['z_test_id'],Ts); tau=np.quantile([(lambda c:c.mean()-(c>t_atc).mean())(rng.choice(ci,len(ci))) for _ in range(500)],0.99)
    for st in SETS:
        z=d['z_'+st]; a=[];b=[]
        for r in range(20):
            idx=rng.choice(len(z),200,replace=False); zb=z[idx]
            k,_=pick(zb,bb,v,s); al=k not in ('clean','color'); a.append(al)
            ct=conf(zb,Ts); b.append(al and (ct.mean()-(ct>t_atc).mean())>tau)
        rows.append(dict(set=st,alarm200=np.mean(a),both200=np.mean(b)))
R=pd.DataFrame(rows); print(R.groupby('set').mean().reindex(SETS).round(2))
