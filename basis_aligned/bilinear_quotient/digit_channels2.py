"""DIGIT CHANNELS 2 -- where in the subspace does each head put its
energy, and where does the content come from one step further back?
528 established two things and broke one. The channel view is
exact (3.57e-7) and its content is nameable: 37.9% of the leading
head's channel content at digit-target queries comes from source
positions that are themselves digits, against a 2.7% base rate.
What broke was the across-heads comparison. Each head's channel
M_h = P^T W_h has effective rank 15.7 of 16, so its column space
IS the whole subspace, and principal angles between column spaces
return 1.000 for every pair by construction. That measured
nothing.
The weighted version is the right object. M_h M_h^T is a 16x16
positive matrix saying how much energy head h can put in each
direction OF the subspace, and normalizing it by its trace removes
the head's overall size -- which 528 showed is what the raw norm
was mostly measuring. Two normalized Gram matrices can then be
compared directly, and the comparison is a genuine one: heads that
emphasize the same directions cooperate, heads that emphasize
different ones partition.
Second, the recursion. The channel content at a source position k
is M v(k) with v(k) = W_v X(k), and X(k) has its own exact writer
decomposition. So the same treatment applies one layer down: which
upstream writers, at the DIGIT source positions identified in 528,
put content into the channel? That is tier 5 of 512, and it is the
first time this program has taken a mechanism two levels back.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the writer decomposition of v(k) reproduces the
      real value vector to 1e-4 relative, checked before scoring;
  (a) COOPERATION: the mean pairwise cosine between normalized
      Gram matrices of the four contributing heads (8.3, 6.1, 6.3,
      8.7) exceeds 0.50, where the cosine is the Frobenius inner
      product of the trace-normalized matrices. This is 528's
      question asked in a form that can fail;
  (b) NOT GENERIC: that mean exceeds the same quantity computed
      for four RANDOMLY chosen heads from the same two layers by
      at least 0.15. Without this, a high cosine could just mean
      all heads look alike in this subspace;
  (c) RECURSION: at the digit source positions, one writer
      supplies at least 30% of the channel content entering the
      value vector, and it is named. If the answer is wte, the
      subspace is carrying raw token identity forward; if it is an
      MLP, something has been computed about the digit.
  NULL: the same recursion measured at NON-digit source positions
      of the same head must give a different leading writer or a
      share at least 10 points lower. If the writer composition is
      identical everywhere, it is a property of the head's value
      map and not of digit content."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_channels2_results.json'
TAG='r.2.0.1'; LAYERS=[6,8]; NFRESH=96; SKIP=2600
QUAD=['8.3','6.1','6.3','8.7']

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
    return orth(torch.cat(vs).T)

def gram_cos(A,B):
    a=A/A.trace().clamp_min(1e-12); b=B/B.trace().clamp_min(1e-12)
    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12))

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    P={li:projector(f'a{li}',probes).float() for li in LAYERS}
    G={}
    for li in LAYERS:
        W=m.transformer.h[li].attn.c_proj.weight.float()
        for h in range(NH):
            M=P[li].T@W[:,h*128:(h+1)*128]
            G[f'{li}.{h}']=M@M.T                      # (16,16)
    # (a) cooperation among the four contributors
    cos=[]
    for i in range(len(QUAD)):
        for j in range(i+1,len(QUAD)):
            c=gram_cos(G[QUAD[i]],G[QUAD[j]])
            cos.append(((QUAD[i],QUAD[j]),round(c,3)))
    meanq=sum(c for _,c in cos)/len(cos)
    print('pairwise normalized-Gram cosines among the four '
          'contributors:',flush=True)
    for pr,c in cos: print(f'   {pr[0]} | {pr[1]}: {c}')
    print(f'   mean {meanq:.3f}',flush=True)
    # (b) the same for random quadruples of heads
    allh=list(G); g=torch.Generator().manual_seed(11); rnd=[]
    for _ in range(20):
        pick=[allh[int(t)] for t in
              torch.randperm(len(allh),generator=g)[:4]]
        cs=[gram_cos(G[pick[i]],G[pick[j]])
            for i in range(4) for j in range(i+1,4)]
        rnd.append(sum(cs)/len(cs))
    meanr=sum(rnd)/len(rnd)
    print(f'   random quadruples of heads: mean {meanr:.3f} '
          f'(min {min(rnd):.3f}, max {max(rnd):.3f})',flush=True)
    # ---- (c) recursion into the value vectors ----
    li,h=int(QUAD[0].split('.')[0]),int(QUAD[0].split('.')[1])
    at=m.transformer.h[li].attn
    Mtop=(P[li].T@at.c_proj.weight.float()[:,h*128:(h+1)*128])
    fresh=cl.fineweb_rows(NFRESH,skip=SKIP)
    cur=fresh[:,:256]
    isdig=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s and s[0].isdigit(): isdig[r,q]=True
    WR=['wte']+[f'{k}{l}' for l in range(li) for k in ('a','m')]
    acc={'digit':{w:0.0 for w in WR},'other':{w:0.0 for w in WR}}
    cnt={'digit':0,'other':0}; errs=[]
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        outs={}; hs=[]
        for lj in range(li):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h2(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
                    return h2
                hs.append(mod.register_forward_hook(mk()))
        cap={}
        hs.append(at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0])))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for hh_ in hs: hh_.remove()
        X=cap['X']
        parts=cl.writer_parts(li,E,outs,'a')
        missing=[w for w in WR if w not in parts]
        if missing:
            print(f'*** missing writers {missing} -- VOID ***')
            json.dump({'void':f'missing {missing}'},
                      open(OUT,'w'),indent=1); return
        tot=sum(parts.values())
        ok,rel=cl.check_parts(parts,X,label=f'a{li} input')
        errs.append(rel)
        s=(X.float().norm(dim=-1,keepdim=True)
           /tot.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        Wv=at.c_v
        full=Wv(X).view(B,T,NH,128)[:,:,h].float()
        cfull=torch.einsum('rd,btd->btr',Mtop,full)
        per={}
        for w in WR:
            vw=Wv((parts[w]*s).to(X.dtype)).view(B,T,NH,128)[:,:,h] \
                .float()
            per[w]=torch.einsum('rd,btd->btr',Mtop,vw)
        recon=sum(per.values())
        errs.append(float((recon-cfull).norm()
                    /cfull.norm().clamp_min(1e-9)))
        for b in range(B):
            r=i+b
            for cls,mask in (('digit',isdig[r]),
                             ('other',~isdig[r])):
                pos=mask.nonzero().squeeze(1)
                if not len(pos): continue
                tot_n=cfull[b,pos].norm(dim=-1).clamp_min(1e-9)
                for w in WR:
                    acc[cls][w]+=float((per[w][b,pos]
                        *cfull[b,pos]).sum(-1).div(tot_n**2).sum())
                cnt[cls]+=len(pos)
    ex=max(errs)
    print(f'\n(0) value-writer reconstruction {ex:.3e}')
    p0=ex<=1e-4
    print(f"(0) {'HELD' if p0 else 'FAILED -- RUN VOID'}")
    if not p0:
        json.dump({'pred_0':False,'exactness':ex},
                  open(OUT,'w'),indent=1); return
    sh={cls:{w:acc[cls][w]/max(cnt[cls],1) for w in WR}
        for cls in acc}
    for cls in ('digit','other'):
        top=sorted(sh[cls],key=lambda w:-sh[cls][w])[:5]
        print(f'  {cls} sources: '+', '.join(
            f'{w} {sh[cls][w]:.3f}' for w in top),flush=True)
    dtop=max(sh['digit'],key=lambda w:sh['digit'][w])
    otop=max(sh['other'],key=lambda w:sh['other'][w])
    va,_=cl.score_bar('a',meanq,0.50)
    vb,_=cl.score_bar('b',meanq-meanr,0.15)
    vc,_=cl.score_bar('c',sh['digit'][dtop],0.30)
    nul=(dtop!=otop) or (sh['digit'][dtop]-sh['other'][dtop]>=0.10)
    print(f"(c) leading writer into the channel at DIGIT sources: "
          f"{dtop} at {sh['digit'][dtop]:.3f}")
    print(f"NULL (non-digit sources differ: leader {otop} at "
          f"{sh['other'][otop]:.3f}): {'ok' if nul else 'VIOLATED'}")
    out={'gram_cosines':cos,'mean_quad':round(meanq,3),
         'mean_random_quad':round(meanr,3),
         'value_writer_shares':{c:{w:round(v,4)
             for w,v in sh[c].items()} for c in sh},
         'digit_leader':dtop,'other_leader':otop,
         'exactness':ex,'pred_0':True,'pred_a':va=='HELD',
         'pred_b':vb=='HELD','pred_c':vc=='HELD',
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
