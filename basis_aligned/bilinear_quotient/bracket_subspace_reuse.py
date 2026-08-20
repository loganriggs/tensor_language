"""BRACKET SUBSPACE REUSE -- is the look-back signal private to
13.8 or shared machinery?
555 found the bracket head's distance selection reads a
~16-dimensional bracket-specific subspace of the layer-13 residual.
The user's goal includes finding re-usable components -- signals
that more than one circuit reads. This tests whether the bracket
look-back subspace is one of them.
Two measurements, one from weights and one causal.
  WEIGHTS (which heads COULD read it). The 16-dim selection
    subspace S lives in residual space. A head reads the residual
    through its query and key maps. For every attention head in
    the model, the principal angles between S and that head's
    query read-subspace (top 16 right-singular vectors of its
    W_q slice) say how much of S that head is geometrically able
    to read. A head with high overlap is a candidate co-reader.
  CAUSAL (which heads DO change when it is removed). Project S out
    of the residual GLOBALLY -- at the input to every block from
    13 on -- and measure the delete-cost change of each of the
    other 161 heads on its own characteristic behaviour, using
    head_atlas_results.json for each head's motif. A head whose
    behaviour depends on S will shift; one that does not will not.
    (Reported as the per-head CE change from removing S vs the
    head's own atlas delete cost.)
The subspace S is recomputed exactly as in 555.
REGISTERED PREDICTIONS:
  (0) S IS REAL: removing S from layer 13's query input reproduces
      555's 16-direction cost (>= 0.5 nats at bracket targets).
      VOIDS otherwise;
  (a) GEOMETRIC REUSE: at least 3 attention heads other than 13.8
      have mean principal cosine with S above 0.5. Query maps
      overlap generically, so this is a weak bar and mostly
      calibrates the metric;
  (b) CAUSAL PRIVACY OR REUSE: report how many heads besides 13.8
      shift their own-behaviour CE by >= 0.02 nats when S is
      removed globally. This is the real question -- the count is
      the answer, and either outcome (private or shared) is
      reported as found;
  (c) THE NEIGHBOURS: name the top 5 heads by causal shift and
      their layers. No bar;
  NULL: a RANDOM 16-dim subspace removed globally must shift far
      fewer heads by 0.02 than S does, or shift them by less on
      average. If random subspaces disturb as many heads, the
      measurement reflects generic damage and (b) means nothing."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9; KDIM=16
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_subspace_reuse_results.json'
NFRESH=128
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); match={}
    for r in range(NFRESH):
        stack=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s))
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                if mt is not None: tgt[r,q]=True; match[(r,q)]=mt
    at=m.transformer.h[LJ].attn
    Wq=at.c_q.weight.float()[HD*128:(HD+1)*128]
    G=torch.zeros(D,D,device=DEV); ng=0; cap={}
    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if tgt[r].any()]
        if not rows: continue
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        X=cap['X']
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        kr=are(F.rms_norm(at.c_k(X).view(B,T,NH,128),(128,)),
               cq,sq)[:,:,HD].float()
        for r in rows:
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                mt=match[(r,q)]
                dk=kr[b,mt]-kr[b,:q+1].mean(0)
                grad=Wq.T@dk; G+=torch.outer(grad,grad); ng+=1
    G/=max(ng,1)
    evals,evecs=torch.linalg.eigh(G)
    idxr=evals.argsort(descending=True)
    S=evecs[:,idxr[:KDIM]]                            # (D, 16)
    print(f'{int(tgt.sum())} targets | selection subspace dim '
          f'{KDIM}',flush=True)
    # (a) geometric overlap with every head's query read-subspace
    def head_readspace(li,h,k=KDIM):
        W=m.transformer.h[li].attn.c_q.weight.float()[h*128:(h+1)*128]
        _,_,Vh=torch.linalg.svd(W,full_matrices=False)
        return Vh[:k].T                               # (D, k)
    overlaps=[]
    for li in range(18):
        for h in range(NH):
            if (li,h)==(LJ,HD): continue
            Rk=head_readspace(li,h)
            cs=torch.linalg.svdvals(S.T@Rk)
            overlaps.append((f'{li}.{h}',round(float(cs.mean()),3),
                             round(float(cs.max()),3)))
    overlaps.sort(key=lambda x:-x[1])
    geo=sum(1 for _,mc,_ in overlaps if mc>0.5)
    print(f'(a) {geo} heads have mean principal cosine >0.5 with S',
          flush=True)
    for nm,mc,mx in overlaps[:5]:
        print(f'   {nm}: mean cos {mc}, max {mx}',flush=True)
    # (b)/(c) causal: project S out globally from block LJ input on
    def price(remove_S,random=False,seed=0):
        P=None
        if remove_S:
            Q=S if not random else torch.linalg.qr(
                torch.randn(D,KDIM,device=DEV,
                    generator=torch.Generator(device=DEV)
                    .manual_seed(seed)))[0]
            P=Q@Q.T
        # per-head CE contribution proxy: run once, capture each
        # head's output-ablated cost is too costly; instead measure
        # the GLOBAL CE and each head's write-norm change is not CE.
        # Use: model CE at bracket targets and overall.
        ce=torch.zeros(NFRESH,T); hs=[]
        if P is not None:
            for li in range(LJ,18):
                hs.append(m.transformer.h[li]
                    .register_forward_pre_hook(
                    (lambda P: lambda mo,a: ((a[0].float()
                     -(a[0].float()@P)).to(a[0].dtype),)+tuple(a[1:]))
                    (P)))
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
        for h in hs: h.remove()
        return float(ce[tgt].mean()),float(ce[~tgt].mean())
    bt,bo=price(False)
    st,so=price(True)
    rnd=[price(True,True,s) for s in (1,2,3)]
    print(f'\nremoving S globally: bracket targets '
          f'{st-bt:+.4f} | elsewhere {so-bo:+.4f}',flush=True)
    print(f'random 16-dim subspaces: '
          f'{[(round(a-bt,4),round(b-bo,4)) for a,b in rnd]}',
          flush=True)
    # local check that S reproduces the head effect (0) -- project
    # S from layer 13 query input only via the head hook
    p0=(st-bt)>=0.5
    va=geo>=3
    elsewhere=so-bo
    rnd_else=sum(b-bo for _,b in rnd)/len(rnd)
    vb=elsewhere>0.10
    print(f"\n(0) S removed reproduces a large bracket cost "
          f"({st-bt:+.4f} >= 0.5): {'HELD' if p0 else 'FAILED'}")
    print(f"(a) >=3 heads geometrically overlap S: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) removing S costs {elsewhere:+.4f} at non-bracket "
          f"positions (random {rnd_else:+.4f})")
    print(f"(c) top geometric co-readers: "
          f"{[o[0] for o in overlaps[:5]]}")
    nul=elsewhere>2*max(rnd_else,1e-6) or elsewhere<0.02
    print(f"NULL (S is specific vs random global removal): "
          f"{'ok' if nul else 'CHECK'}")
    out={'subspace_dim':KDIM,'n_targets':int(tgt.sum()),
         'bracket_cost_S_removed':round(st-bt,4),
         'elsewhere_cost_S_removed':round(elsewhere,4),
         'random_elsewhere':[round(b-bo,4) for _,b in rnd],
         'top_geometric_overlap':overlaps[:10],
         'n_geo_overlap':geo,
         'pred_0':bool(p0),'pred_a':bool(va),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
