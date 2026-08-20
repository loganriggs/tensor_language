"""DIRECTION NAMES -- is the winning stand-in readable?
550 settled the accuracy question and inverted my earlier claim:
projecting a block's write onto its principal directions beats a
lookup table on both cost and description size, by factors of 26
to 100. The table's one remaining advantage was that its columns
could be read -- 542 named block 0's as determiners, punctuation,
initial capitals, digits and sentence openers.
But that naming analysed the table's columns, and the table's
columns ARE the interface basis. The same directions are what the
projection keeps. So the projection may be exactly as readable,
and if it is, the trade-off 550 described disappears: the cheaper
stand-in is also the interpretable one.
This measures readability directly, per block, on clean text.
Each block's interface basis is fitted on the fitting rows. Every
direction is then scored over the vocabulary by the per-token mean
projection, and its top tokens are classified automatically into
determiner, preposition, pronoun, digit, punctuation, capitalized,
space-word or subword. A direction is NAMEABLE when 7 of its 10
top-scoring frequent tokens share a class.
The interesting quantity is not block 0's count -- 542 already
suggested that is high -- but how it changes with depth. Block 0's
input is exactly the token, so token-based naming should work
there. By block 5 the write depends on context that no per-token
score can express, and if nameability collapses, that collapse is
itself the measurement: it says where in the model the
readable-directions method stops working, which is a fact worth
having before anyone builds an interpretation on top of it.
REGISTERED PREDICTIONS:
  (0) DISJOINT: bases fitted only on the fitting rows, verified by
      cl.assert_disjoint. Any overlap VOIDS the run;
  (a) BLOCK 0 IS READABLE: at least 6 of its top 16 directions are
      nameable, extending 542's 6-of-8 to a wider window;
  (b) READABILITY DECAYS: block 5 has at least 4 fewer nameable
      directions than block 0. This is the claim that per-token
      naming is a property of the early layers and not of the
      method;
  (c) THE CURVE: report nameable counts for all six blocks and the
      dominant class of every named direction. No bar;
  NULL: random directions in the same space, 16 of them, yield
      fewer than 3 nameable. If random directions look nameable
      the classifier is too permissive and no count means
      anything."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NB=6; NDIR=16
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'direction_names_results.json'
NFIT=800; NPRICE=96
DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each',
     'every','another','both','all'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through',
      'during','under','against','without','within','onto'}
PRON={'he','she','it','they','we','you','i','him','them','us',
      'me','who','which','what','hers','theirs','itself'}

def fine_class(s):
    t=s.strip().lower()
    if not t: return 'space'
    if t in DET: return 'determiner'
    if t in PREP: return 'preposition'
    if t in PRON: return 'pronoun'
    if t[0].isdigit(): return 'digit'
    if all(not c.isalnum() for c in t): return 'punct'
    if s.strip()[:1].isupper(): return 'capitalized'
    if s.startswith(' '): return 'space_word'
    return 'subword'

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    allrows=cl.rows()
    fit=allrows[:NFIT]; price_rows=allrows[NFIT:NFIT+NPRICE]
    ok,ns=cl.assert_disjoint(fit,price_rows,label='direction_names')
    if not ok:
        json.dump({'pred_0':False,'shared':ns},
                  open(OUT,'w'),indent=1); return
    cnt=torch.bincount(fit.reshape(-1),minlength=V)
    freq=cnt.argsort(descending=True)[:5000]
    st={}
    res={}; examples={}
    for b in range(NB):
        sink=[]
        at=m.transformer.h[b].attn; mlp=m.transformer.h[b].mlp
        h1=at.register_forward_hook(
            lambda mo,i_,o_,b=b: st.__setitem__(
                b,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_,b=b: (sink.append(
                (st[b]+o_.float()).reshape(-1,D)),o_)[1])
        tok=torch.zeros(V,D,device=DEV); tc=torch.zeros(V,device=DEV)
        for i in range(0,NFIT,4):
            bb=fit[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            sink.clear()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            W=sink[0]; flat=idx.reshape(-1)
            tok.index_add_(0,flat,W)
            tc.index_add_(0,flat,torch.ones_like(flat,
                                                 dtype=torch.float))
        h1.remove(); h2.remove()
        tokmean=tok/tc.clamp_min(1).unsqueeze(1)
        mu=(tok.sum(0)/tc.sum())
        # basis from the per-token means, which is what a
        # token-indexed reading of the interface would use
        Yc=tokmean[freq]-mu
        _,_,Vh=torch.linalg.svd(Yc,full_matrices=False)
        named=[]; pure=0
        for k in range(NDIR):
            sc=Yc@Vh[k]
            top=sc.abs().argsort(descending=True)[:10].tolist()
            toks=[cl.d1(int(freq[t])) for t in top]
            cls=[fine_class(s) for s in toks]
            best=max(set(cls),key=cls.count); c=cls.count(best)
            if c>=7: pure+=1
            named.append({'dir':k,'class':best,'purity':c,
                          'tokens':toks[:6]})
        res[b]={'nameable':pure,'directions':named}
        print(f'block {b}: {pure} of {NDIR} directions nameable',
              flush=True)
        for d in named[:4]:
            print(f"   dir {d['dir']}: {d['class']} {d['purity']}/10 "
                  f"-> {[repr(x) for x in d['tokens'][:5]]}",
                  flush=True)
        json.dump({str(k):v for k,v in res.items()},
                  open(OUT,'w'),indent=1)
        del tok,tokmean,Yc
    # NULL: random directions
    g=torch.Generator(device=DEV).manual_seed(11)
    Yc=None
    rnd_pure=0
    tokmean_ref=None
    for k in range(NDIR):
        w=torch.randn(D,generator=g,device=DEV)
        Et=F.rms_norm(m.transformer.wte.weight.float(),(D,))[freq]
        sc=(Et@w)
        top=sc.abs().argsort(descending=True)[:10].tolist()
        cls=[fine_class(cl.d1(int(freq[t]))) for t in top]
        if cls.count(max(set(cls),key=cls.count))>=7: rnd_pure+=1
    counts=[res[b]['nameable'] for b in range(NB)]
    va=counts[0]>=6
    vb=(counts[0]-counts[NB-1])>=4
    nul=rnd_pure<3
    print(f"\n(a) block 0 has {counts[0]} of {NDIR} nameable: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) block 5 has {counts[NB-1]}, at least 4 fewer than "
          f"block 0: {'HELD' if vb else 'FAILED'}")
    print(f"(c) curve across blocks: {counts}")
    print(f"NULL ({rnd_pure} of {NDIR} random directions nameable): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'nameable_by_block':counts,'random_nameable':rnd_pure,
         'blocks':{str(k):v for k,v in res.items()},
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
