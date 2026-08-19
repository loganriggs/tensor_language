"""IOI FLOOR -- capability gate before any task-circuit window
(program rule: floors first; the addition window closed at 0%).
Constructed prompts: "When <A> and <B> went to the <place>, <B>
gave the <object> to" -> correct continuation " <A>". 96 prompts
(8 name pairs x 2 orders x 6 templates), single-token names.
REGISTERED:
  (a) GATE: mean logit margin (IO minus S) and top-1-of-pair
      accuracy reported; the task-circuit window OPENS only if
      pair-accuracy >=60% (chance 50%);
  (b) control: margin on shuffled-name prompts ~0 (|mean| <=0.5)."""
import json, time, itertools, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ioi_floor_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    enc=cl.enc()
    names=[' Mary',' John',' Anna',' Peter',' Sarah',' Tom',
           ' Alice',' Bob']
    ok=[n for n in names if len(enc.encode(n))==1]
    pairs=list(itertools.combinations(ok,2))[:8]
    TEMPL=[('When{A} and{B} went to the store,{B} gave the drink to',''),
           ('When{A} and{B} got home,{B} handed the keys to',''),
           ('After{A} and{B} left the party,{B} gave the coat to',''),
           ('Then{A} and{B} went to the park, and{B} threw the ball to',''),
           ('While{A} and{B} were cooking,{B} passed the salt to',''),
           ('When{A} and{B} finished lunch,{B} gave the bill to','')]
    prompts=[]
    for A,B in pairs:
        for a,b in ((A,B),(B,A)):
            for tpl,_ in TEMPL:
                prompts.append((tpl.replace('{A}',a).replace('{B}',b),
                                a,b))   # correct = a (the IO)
    margins=[]; acc=[]
    ctl=[]
    g=torch.Generator().manual_seed(0)
    for txt,io,s in prompts:
        ids=torch.tensor(enc.encode(txt))[None,:].to(DEV)
        x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()[0,-1]
        ti,ts=enc.encode(io)[0],enc.encode(s)[0]
        margins.append(float(lg[ti]-lg[ts]))
        acc.append(float(lg[ti]>lg[ts]))
        # control: replace both names with random other names
        o1,o2=[ok[i] for i in torch.randperm(len(ok),generator=g)[:2]]
        ctl.append(float(lg[enc.encode(o1)[0]]-lg[enc.encode(o2)[0]]))
    mu=sum(margins)/len(margins); ac=sum(acc)/len(acc)
    mc=sum(ctl)/len(ctl)
    gate=ac>=0.6
    pb=abs(mc)<=0.5
    out={'n_prompts':len(prompts),'mean_margin':round(mu,3),
         'pair_accuracy':round(ac,3),'control_margin':round(mc,3),
         'window_opens':bool(gate),'pred_b':bool(pb),
         'margin_q':[round(sorted(margins)[int(q*len(margins))],2)
                     for q in (0.1,0.5,0.9)]}
    print(f'IOI: acc {ac:.2f} margin {mu:+.2f} control {mc:+.2f}')
    print(f"(a) window opens (acc>=60%): {'YES' if gate else 'NO -- close as capability-absent'}")
    print(f"(b) control margin ~0: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
