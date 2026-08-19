"""Last single-construction published claim: section 217's partial
conservation (private-pair token-damage rho 0.159, 7.5x every random pair,
below the shared baseline 0.229). One span fit per model, one text window.
Sweep: span fits from stats rows 0-60 vs 60-120 (both models), fingerprint
text split into halves (rows 384-416 / 416-448) -- 4 variants; per variant
2 random spans per model give 4 random pairs; shared-pair control per
variant.

REGISTERED PREDICTIONS: (a) private-pair rho above the max random pair in
>= 3/4 variants with rho >= 0.08 there (conservation construction-robust);
(b) private-pair remains BELOW the shared-pair in >= 3/4 variants (the
"partial" in partial conservation is robust too); (c) if (a) fails the
report's conservation sentence gains a ledger row."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
import torch.nn.functional as F
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'conservation_sweep_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def model_prints(name, priv_li, shar_li):
    m2,_=load_elriggs(name, device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def per_token(spec):
        hs=[]
        if spec is not None:
            li,Q,cbar=spec
            def hook(mod,i_,o_):
                c=o_.float().reshape(-1,D)@Q
                return o_-((c-cbar)@Q.T).to(o_.dtype).view_as(o_)
            hs.append(m2.transformer.h[li].mlp.register_forward_hook(hook))
        ces=[]
        for i in range(384,448,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    def span(li, r0, r1, rand=None):
        outs=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(r0,r1,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        Y=torch.cat(outs); mu=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-mu).float(), full_matrices=False)
        if rand is None: Q=orth(Vh[:8].T)
        else:
            g=torch.Generator(device=DEV).manual_seed(rand)
            Q=orth(Vh[8:].T@torch.randn(Vh.shape[0]-8,8,device=DEV,
                                        generator=g))
        return Q, mu.float()@Q
    base=per_token(None)
    P={}
    for tag,(r0,r1) in (('f1',(0,60)),('f2',(60,120))):
        Q,c=span(priv_li,r0,r1); P[f'priv_{tag}']=per_token((priv_li,Q,c))-base
        Q,c=span(shar_li,r0,r1); P[f'shar_{tag}']=per_token((shar_li,Q,c))-base
        for s_ in range(2):
            Q,c=span(priv_li,r0,r1,rand=s_)
            P[f'rnd{s_}_{tag}']=per_token((priv_li,Q,c))-base
    del m2; torch.cuda.empty_cache()
    return P

@torch.no_grad()
def main():
    t0=time.time()
    p18=model_prints('bilin18',6,9)
    p12=model_prints('bilin12',4,6)
    n=len(p18['priv_f1']); half=n//2
    wins=0; below=0; rows=[]
    for ft in ('f1','f2'):
        for w,(a,b) in (('t1',(0,half)),('t2',(half,n))):
            pr=spearman(p18[f'priv_{ft}'][a:b],p12[f'priv_{ft}'][a:b])
            sh=spearman(p18[f'shar_{ft}'][a:b],p12[f'shar_{ft}'][a:b])
            rn=max(spearman(p18[f'rnd{i}_{ft}'][a:b],
                            p12[f'rnd{j}_{ft}'][a:b])
                   for i in range(2) for j in range(2))
            ok=pr>=0.08 and pr>rn
            wins+=ok; below+=(pr<=sh)
            rows.append({'fit':ft,'text':w,'priv':pr,'shar':sh,'rndmax':rn})
            print(f'{ft}/{w}: priv {pr:+.3f} shar {sh:+.3f} rndmax {rn:+.3f}'
                  f'{"  OK" if ok else ""}',flush=True)
    pa=wins>=3; pb=below>=3
    out={'rows':rows,'wins':wins,'below':below,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) conserved in >=3/4: {'HELD' if pa else 'FAILED'} ({wins}/4)")
    print(f"(b) partial (below shared) >=3/4: {'HELD' if pb else 'FAILED'} "
          f"({below}/4)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
