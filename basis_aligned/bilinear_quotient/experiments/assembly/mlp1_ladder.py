"""MLP1 LADDER + ASSEMBLED v2. The crown component is the assembled
dictionary model's dominant cost (+1.216 of +1.850, fold table). Ladder:
(1) fold table (weights-only, published +1.216); (2) empirical token table
(published +0.840); (3) empirical table + rank-32 input-linear residual
map (residual PCA-32 basis P, ridge map X -> R@P, both fit window A --
cost ~74K params); (4) reference: full-rank ridge linear D x D (1.3M).
Then ASSEMBLED v2 with the best rung.

REGISTERED PREDICTIONS: (a) table+rank32 <= +0.55 (halves the empirical
table's cost -- the crown's residual is substantially low-rank-linear in
its input); (b) full linear reference <= +0.35 (mlp1 was the linearization
crown; its whole function is nearly linear-in-input, table+linear should
approach it); (c) assembled v2 total <= +1.50 nats; (d) sanity: rungs 1-2
reproduce published numbers within 10%."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp1_ladder_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def main():
    t0=time.time()
    # fold table (vocab pass)
    tab_fold=torch.zeros(V,D,device=DEV,dtype=torch.float16)
    capm={}
    h1=m.transformer.h[1].mlp.register_forward_hook(
        lambda mo_,i_,o_: capm.__setitem__(1,o_.detach()))
    for s0 in range(0,V,4096):
        idx=torch.arange(s0,min(s0+4096,V),device=DEV)[:,None]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h[:2]:
            x,v1=blk(x,v1,x0)
        tab_fold[s0:s0+idx.shape[0]]=capm[1][:,0].to(torch.float16)
    h1.remove()
    # window A capture: mlp1 outputs + inputs, token ids
    Ys=[]; Xs=[]; Ids=[]
    def cap(mod,i_,o_):
        Ys.append(o_.detach().reshape(-1,D).float())
        Xs.append(i_[0].detach().reshape(-1,D).float())
    h=m.transformer.h[1].mlp.register_forward_hook(cap)
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        m(idx, bb[:,1:].contiguous())
        Ids.append(idx.reshape(-1))
    h.remove()
    Y=torch.cat(Ys); X=torch.cat(Xs); ids=torch.cat(Ids)
    sums=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
    cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
    sums.index_add_(0,ids,Y)
    tab_emp=sums/cnt.clamp_min(1)[:,None]
    tab_emp[cnt==0]=Y.mean(0)
    tab_emp=tab_emp.to(torch.float16)
    Rres=Y-tab_emp[ids].float()
    _,_,Vh=torch.linalg.svd(Rres[:30000], full_matrices=False)
    P=orth(Vh[:32].T)
    lam=1e-2*len(X)
    A=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),
                         X.T@(Rres@P))
    Wfull=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),X.T@Y)
    bfull=Y.mean(0)-X.mean(0)@Wfull
    del Ys,Xs,Y,X,Rres,sums
    cur={}
    def pertok(mode):
        hs=[]
        if mode!='clean':
            def hook(mod,i_,o_):
                idsb=cur['idx']
                if mode=='fold': new=tab_fold[idsb].float()
                elif mode=='emp': new=tab_emp[idsb].float()
                elif mode=='emp32':
                    x=i_[0].float()
                    new=tab_emp[idsb].float()+((x.reshape(-1,D)@A)@P.T)\
                        .view_as(o_).float()
                else:
                    x=i_[0].float().reshape(-1,D)
                    new=(x@Wfull+bfull).view(o_.shape)
                return new.view(o_.shape).to(o_.dtype)
            hs.append(m.transformer.h[1].mlp.register_forward_hook(hook))
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['idx']=idx
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    base=pertok('clean')
    r={}
    for mode in ('fold','emp','emp32','linfull'):
        r[mode]=pertok(mode)-base
        print(f'{mode:8s}: {r[mode]:+.4f}',flush=True)
    pa=r['emp32']<=0.55; pb=r['linfull']<=0.35
    pd=abs(r['fold']-1.216)<=0.13 and abs(r['emp']-0.840)<=0.09
    out={'base':round(base,4),
         'ladder':{k:round(v,4) for k,v in r.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_d':bool(pd)}
    print(f"(a) table+rank32 <=0.55: {'HELD' if pa else 'FAILED'}")
    print(f"(b) full linear <=0.35: {'HELD' if pb else 'FAILED'}")
    print(f"(d) sanity: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
