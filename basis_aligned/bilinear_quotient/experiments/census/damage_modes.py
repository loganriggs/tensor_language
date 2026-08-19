"""DAMAGE MODES -- user direction (SLT-inspired dual): fingerprint DATA
by components, then factor the token x probe damage matrix into
covariance modes -- groups of probes that fail together across data,
and the data that depends on them. Finer than the atlas's hard
clustering (36 comps): probes = 36 components + 72 middle-attention
heads (layers 2-9), damage = per-token CE delta under mean-ablation
(components) / head-deletion (heads), on 12k held-out-window tokens.
Modes that replicate split-half but are NOT explained by the 10-class
taxonomy are NEW supervised circuit labels.
REGISTERED PREDICTIONS:
  (a) STRUCTURE: top-8 modes' energy share >= 2x the within-column
      shuffle null;
  (b) POSITIVE CONTROLS: >= 2 of the top-8 modes align with known
      classes (max class R^2 of mode data-scores >= 0.25) -- the digit
      and induction machinery should appear as modes;
  (c) NEW LABELS: >= 1 top-8 mode with class R^2 <= 0.15 AND split-half
      replication cos >= 0.7 (probe-loading vectors on disjoint token
      halves) -- structure beyond the taxonomy;
  (d) per-mode top probes and sample tokens reported (for naming)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'damage_modes_results.json'
CA=300; NB=12   # 12 batches x 4 rows x 256 = 12288 tokens
MHL=list(range(2,10))

@torch.no_grad()
def main():
    t0=time.time()
    rows=FW[CA:CA+NB*4]
    def ce_vec(hooks):
        ces=[]
        for i in range(0,NB*4,4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return torch.cat(ces)
    base=ce_vec([])
    # component means (for mean-ablation)
    sums={}; hs=[]
    for li in range(18):
        for kind,mod in (('a',m.transformer.h[li].attn),
                         ('m',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; sums[key]=torch.zeros(D,device=DEV)
            def mk(key=key):
                def h(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    sums[key]+=y.detach().float().reshape(-1,D).sum(0)
                return h
            hs.append(mod.register_forward_hook(mk()))
    for i in range(0,NB*4,4):
        bb=rows[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    npos=NB*4*256
    mus={k:v/npos for k,v in sums.items()}
    cols=[]; names=[]
    for li in range(18):
        for kind,mod in (('a',m.transformer.h[li].attn),
                         ('m',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; mu=mus[key]
            if kind=='a':
                def fh(mo,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,mu=mu):
                    return mu.expand_as(o_).to(o_.dtype)
            cols.append(ce_vec([mod.register_forward_hook(fh)])-base)
            names.append(key)
    print(f'{len(names)} component probes done',flush=True)
    # head probes: delete one head's contribution (z slice -> 0)
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    T=256
    for li in MHL:
        at=m.transformer.h[li].attn
        for hd in range(9):
            def fh(mo_,args,out,at=at,hd=hd):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                B=X.shape[0]
                v=at.c_v(X).view(B,T,9,128)
                vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2f=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
                k2f=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
                q2f,k2f=are(q2f,cos,sin),are(k2f,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
                sc2=torch.einsum('bqhd,bkhd->bhqk',q2f.float(),
                                 k2f.float())/128
                pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
                z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                z[:,hd]=0
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X.dtype))
                return (ynew,v1r)
            cols.append(ce_vec([at.register_forward_hook(fh)])-base)
            names.append(f'a{li}h{hd}')
        print(f'head probes L{li} done',flush=True)
    M=torch.stack(cols,1)                   # (Ntok, 108)
    M=M-M.mean(0)
    U,Sg,Vh=torch.linalg.svd(M,full_matrices=False)
    tot=float((Sg**2).sum())
    top8=float((Sg[:8]**2).sum())/tot
    g=torch.Generator(device=DEV).manual_seed(0)
    Msh=torch.stack([M[torch.randperm(M.shape[0],device=DEV,
                       generator=g),j] for j in range(M.shape[1])],1)
    Ssh=torch.linalg.svdvals(Msh)
    top8n=float((Ssh[:8]**2).sum())/float((Ssh**2).sum())
    print(f'top-8 energy {top8:.3f} vs shuffle null {top8n:.3f}',
          flush=True)
    cls=classify(CA,CA+NB*4).reshape(-1).to(DEV)
    Yoh=torch.zeros(len(cls),10,device=DEV)
    Yoh[torch.arange(len(cls)),cls]=1.0
    Me=M[0::2]; Mo=M[1::2]
    _,_,Ve=torch.linalg.svd(Me-Me.mean(0),full_matrices=False)
    _,_,Vo=torch.linalg.svd(Mo-Mo.mean(0),full_matrices=False)
    naligned=0; nnew=0
    report=[]
    toks=rows[:,:256].reshape(-1)
    for k in range(8):
        sc_=U[:,k]*Sg[k]
        r2best=0; cbest=None
        for c in range(10):
            yc=Yoh[:,c]
            r=float(torch.corrcoef(torch.stack([sc_,yc]))[0,1])**2
            if r>r2best: r2best=r; cbest=CLS[c]
        repl=abs(float(Ve[k]@Vo[k]))
        loading=Vh[k]
        topp=[names[i] for i in loading.abs().argsort(descending=True)
              [:5].tolist()]
        ii=sc_.abs().argsort(descending=True)[:6]
        extok=[enc.decode([int(toks[i])]) for i in ii.tolist()]
        report.append({'mode':k,'class_r2':round(r2best,3),
                       'best_class':cbest,'split_half':round(repl,3),
                       'top_probes':topp,'sample_tokens':extok})
        print(f'mode {k}: class-R2 {r2best:.3f} ({cbest}) | split-half '
              f'{repl:.2f} | probes {topp} | tokens {extok}',flush=True)
        if r2best>=0.25: naligned+=1
        if r2best<=0.15 and repl>=0.7: nnew+=1
    pa=top8>=2*top8n; pb=naligned>=2; pc=nnew>=1
    out={'top8_energy':round(top8,3),'null':round(top8n,3),
         'modes':report,'n_aligned':naligned,'n_new':nnew,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) structure >= 2x null: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=2 known-class modes: {'HELD' if pb else 'FAILED'}")
    print(f"(c) >=1 NEW replicable label: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
