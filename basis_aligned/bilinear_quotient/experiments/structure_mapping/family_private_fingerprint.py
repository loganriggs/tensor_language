"""Is the private span the SAME COMPUTATION in both models? Sections 215-216:
each bilinear model has one writer at depth fraction 1/3 whose top-8 output
span no reader shares a vocabulary over. Strongest available probe of what it
IS: per-token CE fingerprints of deleting each private span (bilin18 span
6:1-8, bilin12 span 4:1-8) on the SAME text (rows 384-448), correlated
token-by-token across models.

REGISTERED PREDICTIONS: (a) the private-pair spearman exceeds ALL 9
random-span cross-model pairs and is >= 0.2 (the fraction-1/3 object is
functionally conserved -- it damages the same tokens); (b) instrument
positive control: the matched SHARED pair at fraction 0.5 (bilin18 L9 span
vs bilin12 L6 span) correlates >= 0.2 (text-borne leverage reproduces at
span level); (c) long-shot: private-pair >= shared-pair (the unshared code
is at least as conserved across models as a generic shared one)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
import torch.nn.functional as F
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_private_fingerprint_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def model_prints(name, private_li, shared_li):
    m2,_=load_elriggs(name, device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def per_token(hook_spec):
        hs=[]
        if hook_spec is not None:
            li,Q,cbar=hook_spec
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
    def span_of(li, comp0=0, rand_seed=None):
        outs=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,120,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        Y=torch.cat(outs); mu=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-mu).float(), full_matrices=False)
        if rand_seed is None:
            Q=orth(Vh[comp0:comp0+8].T)
        else:
            g=torch.Generator(device=DEV).manual_seed(rand_seed)
            Q=orth(Vh[8:].T@torch.randn(Vh.shape[0]-8,8,device=DEV,generator=g))
        return Q, mu.float()@Q
    base=per_token(None)
    prints={}
    Q,c=span_of(private_li); prints['private']=per_token((private_li,Q,c))-base
    Q,c=span_of(shared_li);  prints['shared']=per_token((shared_li,Q,c))-base
    for s_ in range(3):
        Q,c=span_of(private_li, rand_seed=s_)
        prints[f'rand{s_}']=per_token((private_li,Q,c))-base
    del m2; torch.cuda.empty_cache()
    return prints

@torch.no_grad()
def main():
    t0=time.time()
    p18=model_prints('bilin18', 6, 9)
    p12=model_prints('bilin12', 4, 6)
    rho_priv=spearman(p18['private'],p12['private'])
    rho_shar=spearman(p18['shared'],p12['shared'])
    rho_rnd=[spearman(p18[f'rand{i}'],p12[f'rand{j}'])
             for i in range(3) for j in range(3)]
    pa=rho_priv>=0.2 and rho_priv>max(rho_rnd)
    pb=rho_shar>=0.2
    pc=rho_priv>=rho_shar
    out={'rho_private':rho_priv,'rho_shared':rho_shar,'rho_random':rho_rnd,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'private pair rho {rho_priv:+.3f} | shared pair {rho_shar:+.3f} | '
          f'random pairs max {max(rho_rnd):+.3f} med '
          f'{sorted(rho_rnd)[4]:+.3f}')
    print(f"(a) private conserved (>=0.2, beats all randoms): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) shared-pair control >=0.2: {'HELD' if pb else 'FAILED'}")
    print(f"(c) long-shot private >= shared: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
