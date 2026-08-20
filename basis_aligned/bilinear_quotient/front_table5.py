"""FRONT TABLE 5 -- the whole table ladder, on clean text.
548 retracted the magnitudes of 542, 544, 546 and 547: the tables
were fitted on the census corpus and priced on cl.fineweb_rows(48),
and 33 of those 48 rows were verbatim in the fitting set. Every
cost was optimistic by an unknown amount. The one clean number in
hand is block 2's, from front_table4 on held-out rows: a pair
table closes about a fifth of the gap where the contaminated run
said two-thirds.
This redoes the whole ladder on a verified split -- tables fitted
on census rows 0-799, priced on rows 800-895, with
cl.assert_disjoint as a gate that must pass before any cost is
reported. Everything else follows 546's rule: each table is fitted
against the ORIGINAL network, not against a partially replaced
one, because independent fits are error-correcting and sequential
fits are error-preserving.
Arms per block (0, 1, 2), each in that block's own 64-dimensional
interface basis, and the basis itself is fitted on the FITTING
rows so it is clean too -- 539-545 fitted their bases on the rows
they priced, which 548 flagged as a milder version of the same
problem:
  unigram   indexed by the current token
  pair      indexed by (previous, current), backing off to unigram
  ceiling   the real write at rank 64
  shuffled  the unigram table with a shuffled index (control)
Then the composition: all three blocks replaced at once by their
best table.
REGISTERED PREDICTIONS:
  (0) DISJOINT: zero priced rows appear in the fitting corpus,
      checked by cl.assert_disjoint. Any overlap VOIDS the run;
  (a) THE TABLES STILL WORK: for each of the three blocks, the
      best table costs less than half of deleting that block's
      write. This is the weakest form of "it is a stand-in" and it
      should survive the correction;
  (b) THE PAIR STILL BEATS THE TOKEN at blocks 1 and 2, by at
      least 0.05 nats each on clean text. 547's direction survived
      the contamination check at block 2 (0.095) and this asks
      whether it holds at block 1 too;
  (c) COMPOSITION: the three-block composition costs at most 1.3
      times the sum of the three individual costs. 546 measured
      additive composition on contaminated text; this is the clean
      version, and the comparison against 545's best rank
      allocation of 1.18 is the benchmark line -- reported with
      both sides' caveats rather than as a headline;
  NULL: at every block the shuffled-index table costs at least
      twice the unigram table. If shuffling is cheap on clean text
      the tables were never using their variables."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; R=64
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'front_table5_results.json'
NFIT=800; NPRICE=96; BLOCKS=[0,1,2]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    V=m.transformer.wte.weight.shape[0]
    allrows=cl.rows()
    fit=allrows[:NFIT]; price_rows=allrows[NFIT:NFIT+NPRICE]
    ok,nshare=cl.assert_disjoint(fit,price_rows,label='front_table5')
    if not ok:
        json.dump({'pred_0':False,'shared':nshare},
                  open(OUT,'w'),indent=1); return
    print(f'(0) DISJOINT: HELD ({NFIT} fit rows, '
          f'{price_rows.shape[0]} priced)',flush=True)
    st={}
    def write_hooks(b,sink):
        at=m.transformer.h[b].attn; mlp=m.transformer.h[b].mlp
        h1=at.register_forward_hook(
            lambda mo,i_,o_,b=b: st.__setitem__(
                b,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_,b=b: (sink.append(
                (st[b]+o_.float()).reshape(-1,D)),o_)[1])
        return [h1,h2]

    def sub_hooks(b,lookup):
        at=m.transformer.h[b].attn; mlp=m.transformer.h[b].mlp
        sl={}
        h1=at.register_forward_hook(
            lambda mo,i_,o_: (sl.__setitem__(
                'a',(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()),o_)[1])
        h2=mlp.register_forward_hook(
            lambda mo,i_,o_: (lookup-sl['a']).to(o_.dtype))
        return [h1,h2]

    def sweep(rows_,b,sink):
        hs=write_hooks(b,sink)
        for i in range(0,rows_.shape[0],4):
            bb=rows_[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()

    TAB={}
    for b in BLOCKS:
        sink=[]; sweep(fit[:200],b,sink)          # basis on FIT rows
        Y=torch.cat(sink); mu=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Y-mu,full_matrices=False)
        B64=Vh[:R]; del sink,Y
        uni=torch.zeros(V,R,device=DEV)
        cnt=torch.zeros(V,device=DEV)
        pidx={}; pac=[]; pcn=[]
        for i in range(0,NFIT,4):
            bb=fit[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            sk=[]; hs=write_hooks(b,sk)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            W=(sk[0]-mu)@B64.T
            cur=idx.reshape(-1)
            prev=torch.cat([torch.zeros(idx.shape[0],1,
                dtype=idx.dtype,device=DEV),idx[:,:-1]],1).reshape(-1)
            uni.index_add_(0,cur,W)
            cnt.index_add_(0,cur,torch.ones_like(cur,
                                                 dtype=torch.float))
            for j,k in enumerate((prev.long()*V+cur.long())
                                 .cpu().tolist()):
                e=pidx.get(k)
                if e is None:
                    pidx[k]=len(pac); pac.append(W[j]); pcn.append(1)
                else:
                    pac[e]=pac[e]+W[j]; pcn[e]+=1
        TAB[b]={'mu':mu,'B':B64,
                'uni':uni/cnt.clamp_min(1).unsqueeze(1),
                'pidx':pidx,
                'P':torch.stack(pac)/torch.tensor(pcn,device=DEV,
                    dtype=torch.float).unsqueeze(1)}
        print(f'block {b}: {int((cnt>0).sum())} tokens, '
              f'{len(pidx)} pairs tabulated',flush=True)

    def lookup(b,idx,pair,shuffle=0):
        t=TAB[b]; Bn=idx.shape[0]; cur=idx.reshape(-1)
        if shuffle:
            g=torch.Generator(device=DEV).manual_seed(shuffle)
            cur=cur[torch.randperm(cur.numel(),generator=g,
                                   device=DEV)]
        C=t['uni'][cur]
        cov=0.0
        if pair:
            prev=torch.cat([torch.zeros(Bn,1,dtype=idx.dtype,
                device=DEV),idx[:,:-1]],1).reshape(-1)
            keys=(prev.long()*V+cur.long()).cpu().tolist()
            sel=torch.tensor([t['pidx'].get(k,-1) for k in keys],
                             device=DEV)
            hit=sel>=0; cov=float(hit.float().mean())
            if int(hit.sum()): C=C.clone(); C[hit]=t['P'][sel[hit]]
        return (t['mu']+C@t['B']).view(Bn,T,D),cov

    def run(spec):
        """spec: list of (block, mode) with mode in
        unigram|pair|ceiling|shuffled|delete"""
        ce=torch.zeros(price_rows.shape[0],T); covs=[]
        for i in range(0,price_rows.shape[0],4):
            bb=price_rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            Bn=bb.shape[0]; hs=[]
            for b,mode in spec:
                if mode=='ceiling':
                    t=TAB[b]; sl={}
                    hs.append(m.transformer.h[b].attn
                        .register_forward_hook(
                        (lambda sl: lambda mo,i_,o_: (sl.__setitem__(
                            'a',(o_[0] if isinstance(o_,tuple)
                                 else o_).detach().float()),o_)[1])(sl)))
                    def fm(mo,i_,o_,t=t,sl=sl):
                        a=sl['a']; w=a+o_.float()
                        new=t['mu']+((w-t['mu'])@t['B'].T)@t['B']
                        return (new-a).to(o_.dtype)
                    hs.append(m.transformer.h[b].mlp
                              .register_forward_hook(fm))
                elif mode=='delete':
                    t=TAB[b]; sl={}
                    hs.append(m.transformer.h[b].attn
                        .register_forward_hook(
                        (lambda sl: lambda mo,i_,o_: (sl.__setitem__(
                            'a',(o_[0] if isinstance(o_,tuple)
                                 else o_).detach().float()),o_)[1])(sl)))
                    hs.append(m.transformer.h[b].mlp
                        .register_forward_hook(
                        (lambda t,sl: lambda mo,i_,o_:
                         (t['mu'].expand_as(o_.float())-sl['a'])
                         .to(o_.dtype))(t,sl)))
                else:
                    lk,cov=lookup(b,idx,mode=='pair',
                                  1 if mode=='shuffled' else 0)
                    covs.append(cov)
                    hs+=sub_hooks(b,lk)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce[i:i+Bn]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').view(Bn,T).cpu()
            for h in hs: h.remove()
        return float(ce.mean()),(sum(covs)/len(covs) if covs else None)

    base,_=run([])
    print(f'\nbaseline CE on held-out rows {base:.4f}',flush=True)
    res={}
    for b in BLOCKS:
        row={}
        for mode in ('delete','ceiling','unigram','pair','shuffled'):
            c,cov=run([(b,mode)])
            row[mode]={'cost':round(c-base,4),
                       'coverage':round(cov,3) if cov else None}
            print(f'block {b} {mode:>8}: {c-base:+.4f}'
                  +(f' (coverage {cov:.3f})' if cov else ''),
                  flush=True)
        res[b]=row
        json.dump({str(k):v for k,v in res.items()},
                  open(OUT,'w'),indent=1)
    best={b:('pair' if res[b]['pair']['cost']<res[b]['unigram']['cost']
             else 'unigram') for b in BLOCKS}
    comp,_=run([(b,best[b]) for b in BLOCKS])
    comp-=base
    ssum=sum(res[b][best[b]]['cost'] for b in BLOCKS)
    va=all(res[b][best[b]]['cost']<0.5*res[b]['delete']['cost']
           for b in BLOCKS)
    vb=all(res[b]['unigram']['cost']-res[b]['pair']['cost']>=0.05
           for b in (1,2))
    vc=comp<=1.3*ssum
    nul=all(res[b]['shuffled']['cost']>=2*res[b]['unigram']['cost']
            for b in BLOCKS)
    print(f"\ncomposition {best}: {comp:+.4f} | sum of parts "
          f"{ssum:+.4f} | ratio {comp/max(ssum,1e-6):.2f}")
    print(f"(a) each table beats half of deleting its block: "
          f"{'HELD' if va else 'FAILED'}")
    print(f"(b) pair beats token at blocks 1 and 2 by >=0.05: "
          f"{'HELD' if vb else 'FAILED'}")
    print(f"(c) composition <= 1.3x sum: "
          f"{'HELD' if vc else 'FAILED'} (545's best rank "
          f"allocation of all six blocks: 1.18)")
    print(f"NULL (shuffled >= 2x unigram everywhere): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'baseline_ce':round(base,4),'n_fit':NFIT,
         'n_priced':int(price_rows.shape[0]),'shared_rows':0,
         'blocks':{str(k):v for k,v in res.items()},
         'best_mode':{str(k):v for k,v in best.items()},
         'composition':round(comp,4),'sum_parts':round(ssum,4),
         'pred_0':True,'pred_a':bool(va),'pred_b':bool(vb),
         'pred_c':bool(vc),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
