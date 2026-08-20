"""DIGIT CHANNELS -- go below the head, and combine across heads.
521 concluded the digit subspace is "distributed": four heads
across two layers share it, no single head carries more than a
third, and individual removals recover 64% of the joint. That
conclusion is only forced if HEADS are the units. They are not.
The subspace is 16 directions in residual space. Head h writes
into the residual through W_proj restricted to its own 128
columns, so its channel into the subspace is the 16x128 matrix
    M_h = P^T W_proj[:, h*128:(h+1)*128]
and everything that head can ever contribute to the subspace
passes through it. M_h has rank at most 16 out of 128, so each
head's write into the subspace is a LOW-DIMENSIONAL channel, and
the natural unit is that channel rather than the head.
This run does three things the head-level analysis could not.
  WEIGHTS ONLY. M_h is computable from weights with no data at
    all. Its singular values say which heads even have a channel
    and how wide it is. That is an advance prediction of the
    causal ranking 521 measured, made from the weights alone --
    the same shape of claim as the head atlas naming the newline
    head before any newline was measured.
  SHARED OR PARTITIONED. The four contributing heads each write
    into a subspace of the same 16 directions. Do they write into
    the SAME directions -- cooperating, which would explain the
    superadditivity -- or into different ones, partitioning the
    subspace? Principal angles between the column spaces of the
    M_h answer this exactly, again from weights only.
  SOURCE STRUCTURE. Inside a head, the channel content is
    z_h = SUM_k score(q,k) v_h(k), so the channel is fed by
    specific SOURCE POSITIONS. At digit-target queries, which
    positions supply it? If the answer is "earlier digits", the
    subspace is carrying digit content forward and the mechanism
    is nameable.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the per-head channel projections sum to the
      full-bundle projection of the component output, to 1e-4
      relative. Failure VOIDS the run;
  (a) THE WEIGHTS KNEW: across the 18 heads, the Frobenius norm of
      M_h rank-correlates with the per-head causal effects
      measured in 521 (8.3 +0.0299, 6.1 +0.0236, 6.3 +0.0099,
      8.7 +0.0087, the rest at or below 0.0027) with Spearman
      rho >= 0.60;
  (b) NARROW CHANNELS: the leading head's channel has effective
      rank <= 8 of the 16 available (participation ratio of its
      singular values), i.e. a head writes into a proper part of
      the subspace rather than all of it;
  (c) COOPERATION, NOT PARTITION: the mean principal cosine
      between the channel column spaces of the two leading heads
      is >= 0.50. This is a directional bet -- superadditive
      heads that share directions cooperate; orthogonal channels
      would mean they partition the subspace and the
      superadditivity needs a different explanation. Either
      answer is reported.
  (d) SOURCE: at digit-target queries, the share of channel
      content supplied by source positions whose token is a digit
      is at least 2x the share of digits among positions in the
      same contexts.
  NULL: (a) is recomputed against a RANDOM 16-dimensional
      subspace in the same components -- its channel norms must
      NOT predict the measured digit effects (rho < 0.30). If a
      random subspace predicts just as well, the channel norm is
      measuring head size and not this subspace."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_channels_results.json'
TAG='r.2.0.1'; LAYERS=[6,8]; NFRESH=96; SKIP=2400
MEASURED={'8.3':0.0299,'6.1':0.0236,'6.3':0.0099,'8.7':0.0087,
          '6.7':0.0024,'8.6':0.0022,'8.4':0.0014,'8.2':0.0011}

def spearman(a,b):
    def rank(v):
        o=sorted(range(len(v)),key=lambda i:v[i])
        r=[0.0]*len(v)
        for pos,i in enumerate(o): r[i]=pos
        return r
    ra,rb=rank(a),rank(b); n=len(a)
    ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    da=sum((x-ma)**2 for x in ra)**0.5
    db=sum((y-mb)**2 for y in rb)**0.5
    return num/max(da*db,1e-9)

def projector(key,probes):
    seen=set(); keep=[]
    for pr in probes:
        _,k,stag,blk=pr
        if k!=key or (k,stag,tuple(blk)) in seen: continue
        seen.add((k,stag,tuple(blk))); keep.append(pr)
    drop=set()
    for i,(_,k1,s1,b1) in enumerate(keep):
        for j,(_,k2,s2,b2) in enumerate(keep):
            if i!=j and k1==k2 and s1==s2 and b2[0]<=b1[0] \
               and b1[1]<=b2[1] and (b1[1]-b1[0])<(b2[1]-b2[0]):
                drop.add(i)
    vs=[cl.pca_block(p[1],p[2],p[3])
        for i,p in enumerate(keep) if i not in drop]
    return orth(torch.cat(vs).T)          # (D, r)

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    P={li:projector(f'a{li}',probes).float() for li in LAYERS}
    for li in LAYERS:
        print(f'a{li} projector rank {P[li].shape[1]}',flush=True)
    # ---- weights-only channel analysis ----
    chan={}; Ms={}
    for li in LAYERS:
        W=m.transformer.h[li].attn.c_proj.weight.float()   # (D, NH*128)
        for h in range(NH):
            Wh=W[:,h*128:(h+1)*128]
            M=P[li].T@Wh                       # (r, 128)
            sv=torch.linalg.svdvals(M)
            pr=float((sv.sum()**2)/(sv.pow(2).sum().clamp_min(1e-12)))
            chan[f'{li}.{h}']={'fro':round(float(M.norm()),4),
                               'top_sv':round(float(sv[0]),4),
                               'eff_rank':round(pr,2),
                               'sv':[round(float(x),4) for x in sv[:6]]}
            Ms[f'{li}.{h}']=M
    for k in sorted(chan,key=lambda k:-chan[k]['fro'])[:8]:
        c=chan[k]
        print(f"  {k}: |M| {c['fro']:.3f} top-sv {c['top_sv']:.3f} "
              f"eff-rank {c['eff_rank']}",flush=True)
    heads=[k for k in chan]
    meas=[MEASURED.get(k,0.0) for k in heads]
    fro=[chan[k]['fro'] for k in heads]
    rho=spearman(fro,meas)
    # NULL: random subspace of the same rank
    rnd_rho=[]
    for s in (3,11,19):
        g=torch.Generator(device=DEV).manual_seed(s)
        rf=[]
        for li in LAYERS:
            V=torch.randn(P[li].shape[1],D,generator=g,device=DEV)
            Q=orth(V.T).float()
            W=m.transformer.h[li].attn.c_proj.weight.float()
            for h in range(NH):
                rf.append(float((Q.T@W[:,h*128:(h+1)*128]).norm()))
        rnd_rho.append(round(spearman(rf,meas),3))
    # ---- principal angles between the two leading channels ----
    order=sorted(heads,key=lambda k:-chan[k]['fro'])
    top2=order[:2]
    def colspace(M):
        U,S,_=torch.linalg.svd(M,full_matrices=False)
        keep=(S>0.05*S[0]).sum().item()
        return U[:,:max(keep,1)]
    A=colspace(Ms[top2[0]]); Bm=colspace(Ms[top2[1]])
    cs=torch.linalg.svdvals(A.T@Bm)
    meancos=float(cs.mean())
    print(f'\nprincipal cosines between {top2[0]} and {top2[1]}: '
          f'{[round(float(x),3) for x in cs[:6]]} mean {meancos:.3f}',
          flush=True)
    # also across the four measured contributors
    quad=['8.3','6.1','6.3','8.7']
    pairs={}
    for i in range(len(quad)):
        for j in range(i+1,len(quad)):
            a,b=quad[i],quad[j]
            if a not in Ms or b not in Ms: continue
            c=torch.linalg.svdvals(colspace(Ms[a]).T@colspace(Ms[b]))
            pairs[f'{a}|{b}']=round(float(c.mean()),3)
    print(f'  pairwise mean cosines among the four contributors: '
          f'{pairs}',flush=True)
    # ---- source structure at digit targets ----
    fresh=cl.fineweb_rows(NFRESH,skip=SKIP)
    nxt=fresh[:,1:257]; cur=fresh[:,:256]
    dig=torch.zeros(NFRESH,T,dtype=torch.bool)
    isdig=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            t=cl.d1(int(nxt[r,q])).strip()
            if t and t[0].isdigit(): dig[r,q]=True
            s=cl.d1(int(cur[r,q])).strip()
            if s and s[0].isdigit(): isdig[r,q]=True
    base_rate=float(isdig.float().mean())
    li,h=int(order[0].split('.')[0]),int(order[0].split('.')[1])
    at=m.transformer.h[li].attn
    Mtop=Ms[order[0]]
    src={'digit':[0.0,0],'other':[0.0,0]}
    exact=[]
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        cap={}
        hs=[at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0])),
            at.register_forward_hook(
            lambda mo_,a_,o_: cap.__setitem__('v1',a_[1]))]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for hh_ in hs: hh_.remove()
        X=cap['X']
        z,vm=cl.head_parts(li,X,cap.get('v1'))
        # channel content per (query, source): score(q,k) * M v(k)
        Wq=at.c_proj.weight.float()[:,h*128:(h+1)*128]
        vv=vm[:,:,h].float()                       # (B,T,128)
        cv=torch.einsum('rd,btd->btr',Mtop,vv)     # (B,T,r)
        # attention scores for this head
        are=__import__('sys').modules[
            type(at).__module__].apply_rotary_emb
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r2(W):
            return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                       cq,sq)[:,:,h].float()
        s1=torch.einsum('bqd,bkd->bqk',r2(at.c_q),r2(at.c_k))/128
        s2=torch.einsum('bqd,bkd->bqk',r2(at.c_q2),r2(at.c_k2))/128
        sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
        contrib=torch.einsum('bqk,bkr->bqkr',sc,cv).norm(dim=-1)
        # exactness: summed channel content must equal M z_h
        lhs=torch.einsum('bqk,bkr->bqr',sc,cv)
        rhs=torch.einsum('rd,bqd->bqr',Mtop,z[:,h].float())
        exact.append(float((lhs-rhs).norm()/rhs.norm().clamp_min(1e-9)))
        cc=contrib.cpu()
        for b in range(B):
            r=i+b
            for q in dig[r].nonzero().squeeze(1).tolist():
                tot=float(cc[b,q,:q+1].sum())
                if tot<=0: continue
                dmask=isdig[r,:q+1]
                src['digit'][0]+=float(cc[b,q,:q+1][dmask].sum())/tot
                src['digit'][1]+=1
                src['other'][0]+=float(
                    cc[b,q,:q+1][~dmask].sum())/tot
                src['other'][1]+=1
    ex=max(exact)
    print(f'\n(0) channel decomposition exactness {ex:.3e}')
    p0=ex<=1e-4
    dshare=src['digit'][0]/max(src['digit'][1],1)
    print(f"(0) {'HELD' if p0 else 'FAILED -- RUN VOID'}")
    if not p0:
        json.dump({'pred_0':False,'exactness':ex},
                  open(OUT,'w'),indent=1); return
    va,_=cl.score_bar('a',rho,0.60)
    vb,_=cl.score_bar('b',8-chan[order[0]]['eff_rank'],1e-9)
    vc,_=cl.score_bar('c',meancos,0.50)
    vd,_=cl.score_bar('d',dshare-2*base_rate,1e-9)
    nul=max(abs(x) for x in rnd_rho)<0.30
    print(f'  (a) Spearman(|M_h|, measured effect) = {rho:.3f}')
    print(f"  (b) leading head {order[0]} effective rank "
          f"{chan[order[0]]['eff_rank']} of {P[LAYERS[0]].shape[1]}")
    print(f'  (d) digit-source share of channel content {dshare:.4f} '
          f'vs digit base rate {base_rate:.4f}')
    print(f"  NULL random-subspace rhos {rnd_rho}: "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'channels':chan,'spearman_weights_vs_causal':round(rho,3),
         'random_subspace_rhos':rnd_rho,
         'top_two':top2,'mean_principal_cosine':round(meancos,3),
         'pairwise_cosines':pairs,
         'digit_source_share':round(dshare,4),
         'digit_base_rate':round(base_rate,4),
         'exactness':ex,'pred_0':True,'pred_a':va=='HELD',
         'pred_b':vb=='HELD','pred_c':vc=='HELD','pred_d':vd=='HELD',
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
