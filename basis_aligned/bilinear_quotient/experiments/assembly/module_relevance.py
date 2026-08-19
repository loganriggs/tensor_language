"""MODULE RELEVANCE TABLE -- uniform per-component importance for the
benchmark figures: replace each of the 36 components (18 attention
outputs, 18 MLP outputs) with its mean output (computed on window A),
one at a time, and measure CE cost on window C. Also computes the
CEILING: the wte-only model (all blocks removed -- logits straight from
the normalized embedding), which anchors 0% functionality.
No registered bars -- this is an instrument run for the figures; the
per-layer decline pattern is already certified in the flat track."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'module_relevance_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def evalCE(hooks=()):
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    return float(torch.cat(ces).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base=evalCE()
    # ceiling: logits from normalized embeddings only
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,))
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    ceiling=float(torch.cat(ces).mean())
    print(f'base {base:.4f} | wte-only ceiling {ceiling:.4f}',flush=True)
    # means on window A (one pass, all 36 accumulators)
    sums={}; cnt=0
    hs=[]
    for li in range(18):
        for kind,mod in (('attn',m.transformer.h[li].attn),
                         ('mlp',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; sums[key]=torch.zeros(D,device=DEV)
            def mk(key=key):
                def h(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    sums[key]+=y.detach().float().reshape(-1,D).sum(0)
                return h
            hs.append(mod.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        cnt+=4*256
    for h in hs: h.remove()
    means={k:(v/cnt) for k,v in sums.items()}
    out={'base':round(base,4),'ceiling':round(ceiling,4),'mean_abl':{}}
    for li in range(18):
        for kind,mod in (('attn',m.transformer.h[li].attn),
                         ('mlp',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; mu=means[key]
            if kind=='attn':
                def fh(mo,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,mu=mu):
                    return mu.expand_as(o_).to(o_.dtype)
            h=mod.register_forward_hook(fh)
            c=evalCE()-base
            h.remove()
            out['mean_abl'][key]=round(c,4)
            print(f'{key:7s} {c:+.4f}',flush=True)
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
