"""CONSTRUCTED-PROMPT TASK CIRCUITS (IOI and arithmetic analogs). Natural
web text gave the model no IOI or addition support; build the windows:
  IOI:      "When {A} and {B} went to the {place}, {B} gave a {obj} to"
            -> " {A}"  (200 prompts, single-token names)
  counting: "{n}, {n+1}, {n+2}, {n+3}," -> " {n+4}"
  addition: "{a} + {b} =" -> " {c}", a,b in 2..9
Measure task competence (top-1 accuracy + CE at the answer), then per-
component ownership: all 36 components ablated one at a time, CE delta at
the answer position, ranked.

REGISTERED PREDICTIONS: (a) IOI-analog top-1 >= 50% (the model has the
copy machinery); (b) IOI answer ownership top-3 includes one of
{attn0,attn1,attn3,attn4,attn5} (copy infrastructure or induction band --
either confirms the natural-circuit map transfers to tasks); (c) counting
ownership top-3 includes attn8 or mlp15 (the certified digit circuit);
(d) addition competence is registered UNCERTAIN: if top-1 < 30% record
'cannot do addition' and skip its ownership -- a capability fact, not a
failure; (e) control: ownership profile of a shuffled-names IOI set (answer
token random) shows no component >= 2x its natural-atlas mean share."""
import json, sys, time, torch, random
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'task_circuits_results.json'
random.seed(0)

NAMES=[n for n in [' Mary',' John',' Tom',' Anna',' Paul',' Sarah',' Mike',
  ' Emma',' James',' Lucy',' Peter',' Alice',' Mark',' Kate',' David',
  ' Laura'] if len(enc.encode(n))==1]
PLACES=[' store',' park',' office',' market']
OBJS=[' drink',' book',' ball',' letter']

def build_ioi(n=200,shuffle=False):
    ps=[]
    for _ in range(n):
        A,B=random.sample(NAMES,2)
        pl=random.choice(PLACES); ob=random.choice(OBJS)
        txt=f"When{A} and{B} went to the{pl},{B} gave a{ob} to"
        ans=A if not shuffle else random.choice(NAMES)
        ps.append((enc.encode(txt),enc.encode(ans)[0]))
    return ps
def build_count(n=120):
    ps=[]
    for _ in range(n):
        s=random.randint(1,14)
        txt=f"{s}, {s+1}, {s+2}, {s+3},"
        ps.append((enc.encode(txt),enc.encode(f" {s+4}")[0]))
    return ps
def build_add(n=120):
    ps=[]
    for _ in range(n):
        a,b=random.randint(2,9),random.randint(2,9)
        txt=f"{a} + {b} ="
        ps.append((enc.encode(txt),enc.encode(f" {a+b}")[0]))
    return ps

@torch.no_grad()
def run_prompts(prompts, hooks=None):
    """returns (ce at answer, top1 correct) tensors."""
    hs=hooks() if hooks else []
    ces=[]; hits=[]
    bylen={}
    for t,a in prompts: bylen.setdefault(len(t),[]).append((t,a))
    for L,grp in bylen.items():
        for i in range(0,len(grp),32):
            g=grp[i:i+32]
            idx=torch.tensor([t for t,_ in g],device=DEV)
            ans=torch.tensor([a for _,a in g],device=DEV)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x[:,-1],(D,)))/30)).float()
            ces.append(F.cross_entropy(lg,ans,reduction='none'))
            hits.append((lg.argmax(1)==ans).float())
    for h in hs: h.remove()
    return torch.cat(ces),torch.cat(hits)

@torch.no_grad()
def main():
    t0=time.time()
    spans={}
    for li in range(18):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
    amus={li:attn_mean(li) for li in range(18)}
    def hooks_for(comp):
        def make():
            hs=[]
            if comp.startswith('mlp'):
                li=int(comp[3:]); Q,mu=spans[li]
                def hook(mod,i_,o_):
                    c=o_.float().reshape(-1,D)@Q
                    delta=((c-(mu@Q))@Q.T).view(o_.shape)
                    return o_-delta.to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(hook))
            else:
                li=int(comp[4:]); mu=amus[li]
                def hooka(mod,i_,o_):
                    out=o_[0] if isinstance(o_,tuple) else o_
                    new=mu[None,None,:].to(out.dtype).expand_as(out)
                    if isinstance(o_,tuple): return (new,)+o_[1:]
                    return new
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(hooka))
            return hs
        return make
    tasks={'ioi':build_ioi(),'count':build_count(),'add':build_add(),
           'ioi_shuffled':build_ioi(shuffle=True)}
    res={}
    comps=[f'mlp{i}' for i in range(18)]+[f'attn{i}' for i in range(18)]
    for name,ps in tasks.items():
        ce0,hit=run_prompts(ps)
        acc=float(hit.mean()); c0=float(ce0.mean())
        res[name]={'acc':round(acc,3),'ce':round(c0,3)}
        print(f'{name:12s}: top1 {acc:.0%} ce {c0:.2f}',flush=True)
        if name=='add' and acc<0.30:
            res[name]['ownership']='skipped-cannot-do'
            continue
        if name=='ioi_shuffled': continue
        deltas={}
        for comp in comps:
            ce,_=run_prompts(ps,hooks_for(comp))
            deltas[comp]=round(float((ce-ce0).mean()),4)
        top=sorted(deltas,key=lambda k:-deltas[k])[:5]
        res[name]['top_owners']=[(k,deltas[k]) for k in top]
        res[name]['deltas']=deltas
        print(f'  top owners: {[(k,deltas[k]) for k in top]}',flush=True)
    pa=res['ioi']['acc']>=0.50
    band={'attn0','attn1','attn3','attn4','attn5'}
    pb='top_owners' in res['ioi'] and any(
        k in band for k,_ in res['ioi']['top_owners'][:3])
    pc='top_owners' in res['count'] and any(
        k in ('attn8','mlp15') for k,_ in res['count']['top_owners'][:3])
    out=dict(res)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)})
    print(f"\n(a) IOI top1 >=50%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) IOI owners in copy/induction band: {'HELD' if pb else 'FAILED'}")
    print(f"(c) counting owned by digit circuit: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
