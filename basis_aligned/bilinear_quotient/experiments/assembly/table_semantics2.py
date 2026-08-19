"""TABLE GROUNDING v2 -- centered instrument. v1 found fold-vs-empirical
median cosine 0.917 but the shuffled null sat at 0.628: all rows share a
large common mean direction, so raw cosine measures the shared mean, not
row identity. v2 subtracts the mean empirical row (and mean fold row)
before comparing.
REGISTERED PREDICTIONS:
  (a) centered median cosine >= 0.5, centered shuffled null <= 0.10;
  (b) the class-structure gap survives centering (>= 3x shuffled null)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'table_semantics2_results.json'
CA,CB=300,512

@torch.no_grad()
def main():
    t0=time.time()
    # fold table: every vocab token alone through block 0, capture mlp0
    fold=torch.zeros(V,D,device=DEV)
    cap=[]
    h=m.transformer.h[0].mlp.register_forward_hook(
        lambda mo,i_,o_: cap.append(o_.detach().float().reshape(-1,D)))
    for s0 in range(0,V,4096):
        idx=torch.arange(s0,min(s0+4096,V),device=DEV)[:,None]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
        m.transformer.h[0](x,None,x0)
        fold[s0:s0+idx.shape[0]]=cap.pop()
    h.remove()
    # empirical table on window A (clean model)
    sums=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
    cap2=[]
    h=m.transformer.h[0].mlp.register_forward_hook(
        lambda mo,i_,o_: cap2.append(o_.detach().float().reshape(-1,D)))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        ids=bb[:,:-1].reshape(-1)
        Y=cap2.pop()
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        sums.index_add_(0,ids,Y)
    h.remove()
    ok=cnt>=5
    emp=sums[ok]/cnt[ok][:,None]; fo=fold[ok]
    emp=emp-emp.mean(0); fo=fo-fo.mean(0)
    cos=F.cosine_similarity(emp,fo,dim=1)
    med=float(cos.median())
    g=torch.Generator(device=DEV).manual_seed(0)
    perm=torch.randperm(fo.shape[0],device=DEV,generator=g)
    cosn=F.cosine_similarity(emp,fo[perm],dim=1)
    medn=float(cosn.median())
    print(f'tokens with >=5 occurrences: {int(ok.sum())}')
    print(f'(1) fold-vs-empirical median cosine {med:.3f} '
          f'(shuffled null {medn:.3f})',flush=True)
    # semantic classes over the well-observed tokens
    toks=torch.nonzero(ok).squeeze(1).tolist()
    def tclass(t):
        s=enc.decode([t])
        st=s.strip()
        if st.isdigit(): return 0
        if st and not any(c.isalnum() for c in st): return 1
        if s.startswith(' ') and st[:1].isalpha(): return 2
        if st.isalpha(): return 3
        return 4
    lab=torch.tensor([tclass(t) for t in toks],device=DEV)
    E=emp/emp.norm(dim=1,keepdim=True).clamp_min(1e-8)
    C=E@E.T
    def gap(l):
        same=(l[:,None]==l[None,:]); eye=torch.eye(len(l),device=DEV,
                                                   dtype=torch.bool)
        w=float(C[same&~eye].mean()); b=float(C[~same].mean())
        return w-b,w,b
    gp,w,b=gap(lab)
    perm2=lab[torch.randperm(len(lab),device=DEV,generator=g)]
    gpn,_,_=gap(perm2)
    print(f'(2) within-vs-between class cosine gap {gp:.3f} '
          f'(within {w:.3f}, between {b:.3f}; shuffled null {gpn:.3f})')
    names=['digit','punct','word-start','subword','other']
    percls={}
    for k in range(5):
        lk=lab==k
        if lk.sum()>4:
            sub=C[lk][:,lk]
            eye=torch.eye(int(lk.sum()),device=DEV,dtype=torch.bool)
            percls[names[k]]=round(float(sub[~eye].mean()),3)
    print(f'per-class within-cosine: {percls}')
    pa=med>=0.5 and medn<=0.10
    pb=gp>=3*max(abs(gpn),1e-3)
    out={'n_tokens':int(ok.sum()),'median_cos':round(med,3),
         'null_cos':round(medn,3),'class_gap':round(gp,3),
         'class_gap_null':round(gpn,3),'per_class':percls,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) centered fold matches empirical (>=0.5, null<=0.10): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) class structure >= 3x null: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
