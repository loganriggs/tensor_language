"""CLASS SPARING -- 464: confidence does NOT explain the
punctuation sparing (463): matching positions on the intact
model's top-1 probability leaves 91.4% of it intact (-0.0229 of
-0.0251), and damage is roughly FLAT across confidence deciles
rather than falling. So after a circuit reading, a frequency
reading and a selection artifact all died, something genuinely
class-specific survives.
Scope it: is PUNCTUATION special, or is it one member of a
structural/format family? Price the same 16-dim bundle ablation
on fresh FineWeb rows for six target classes, each against
everything else, using the mechanical class definitions the
program already uses.
Classes: punct, newline, digit, subword (no leading space,
alphabetic), space_word, capitalized.
REGISTERED PREDICTIONS:
  (a) FAMILY, NOT SINGLETON: at least one class besides
      punctuation shows negative dissociation (sparing) --
      newline is the natural candidate;
  (b) CONTENT CLASSES PAY: space_word and capitalized show
      POSITIVE dissociation (extra damage), i.e. the sparing is
      not universal;
  (c) RANK: punctuation's sparing is the largest, or within 0.005
      of the largest."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'class_sparing_results.json'
TAG='r.13.2.1'
NFRESH=48

def classify(tok):
    s=cl.d1(int(tok)); st=s.strip()
    return {'punct':bool(st) and not any(c.isalnum() for c in st),
            'newline':chr(10) in s,
            'digit':st.isdigit(),
            'subword':(not s.startswith(' ')) and st.isalpha(),
            'space_word':s.startswith(' ') and st.isalpha(),
            'capitalized':s.startswith(' ') and bool(st)
                          and st[:1].isupper()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    KINDS=['punct','newline','digit','subword','space_word',
           'capitalized']
    masks={k:torch.zeros(NFRESH,T,dtype=torch.bool) for k in KINDS}
    for r in range(NFRESH):
        for q in range(T):
            c=classify(int(fresh[r,q+1]))
            for k in KINDS: masks[k][r,q]=c[k]
    print({k:int(masks[k].sum()) for k in KINDS},flush=True)
    dce=torch.zeros(NFRESH,T)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        def fwd(ab):
            hs=cl.proj_hooks(cl.leaf(TAG)['top_probes']) if ab \
                else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                              reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            return c
        dce[i:i+4]=fwd(True)-fwd(False)
        print(f'batch {i} done',flush=True)
    dd=dce.reshape(-1)
    res={}
    for k in KINDS:
        mk=masks[k].reshape(-1)
        if int(mk.sum())<20: 
            res[k]={'n':int(mk.sum()),'note':'too few'}
            continue
        inm=float(dd[mk].mean()); out_=float(dd[~mk].mean())
        res[k]={'n':int(mk.sum()),'dce_in':round(inm,4),
                'dce_out':round(out_,4),
                'dissociation':round(inm-out_,4)}
        print(f"{k}: n {res[k]['n']} in {inm:+.4f} out {out_:+.4f}"
              f" dissoc {inm-out_:+.4f}",flush=True)
    ok=[k for k in KINDS if 'dissociation' in res[k]]
    spared=[k for k in ok if res[k]['dissociation']<0]
    pa=len([k for k in spared if k!='punct'])>=1
    pb=all(res[k]['dissociation']>0 for k in
           ('space_word','capitalized') if k in ok)
    best=min(ok,key=lambda k:res[k]['dissociation'])
    pc=(res['punct']['dissociation']
        <=res[best]['dissociation']+0.005)
    out={'classes':res,'spared_classes':spared,
         'most_spared':best,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f'spared: {spared} | most spared: {best}')
    for nm,v in (('a','a second class is spared (family)'),
                 ('b','content classes pay extra damage'),
                 ('c','punctuation is the most spared')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
