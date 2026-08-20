"""HEAD 12.6 CLASS POTENCY -- 492: the causal test (491) supports
the structure reading but not at the strength I registered. On
structured rows at match positions, blocking 12.6's distant reads
to punctuation and newline positions reproduces 44.4% of the
4-token-window damage, against 21.9% for a control blocking an
EXACTLY equal number of distant prose positions (3525 vs 3525).
Structure carries twice what content carries; neither dominates,
and about a third of the damage comes from distant positions in
neither class.
Decompose it properly rather than guessing which class to add.
Block each token class in turn (punctuation, newline, digit,
capitalised, space-word, subword) at distant positions and measure
both the damage and the number of positions blocked, giving a
DAMAGE-PER-BLOCKED-POSITION potency that is comparable across
classes of very different sizes.
This also cross-checks two independent measurements: 489 ranked
these classes by how ENRICHED 12.6's reads are for them; this
ranks them by how much BLOCKING them costs. Those orderings should
agree if the enrichment means what we think.
REGISTERED PREDICTIONS:
  (a) PUNCTUATION IS MOST POTENT: punctuation has the highest
      damage per blocked position of the six classes;
  (b) PROSE IS LEAST: subword content has the lowest;
  (c) CONVERGENCE: the potency ranking correlates with 489's
      enrichment ranking at Spearman >= 0.5."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_12_6_classes_results.json'
NFRESH=48
ENRICH={'punct':2.33,'capitalized':1.79,'digit':1.33,
        'newline':1.27,'space_word':0.68,'subword':0.38}

def cls_of(tok):
    s=cl.d1(int(tok)); st=s.strip()
    return {'punct':bool(st) and not any(c.isalnum() for c in st),
            'newline':chr(10) in s,'digit':st.isdigit(),
            'capitalized':s.startswith(' ') and bool(st)
                          and st[:1].isupper(),
            'space_word':s.startswith(' ') and st.isalpha(),
            'subword':(not s.startswith(' ')) and st.isalpha()}

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    def isstruct(t):
        c=cls_of(t); return c['punct'] or c['newline']
    dens=torch.tensor([sum(isstruct(int(fresh[r,q]))
                           for q in range(T))/T
                       for r in range(NFRESH)])
    keep=(dens>=float(dens.median())).nonzero().squeeze(1).tolist()
    KINDS=list(ENRICH)
    blocked={k:0 for k in KINDS}
    def run(mode):
        tot=0.0; n=0
        for i in range(0,len(keep),4):
            rid=torch.tensor(keep[i:i+4])
            bb=fresh[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=len(rid); hs=[]
            if mode is not None:
                bm=torch.zeros(B,T,dtype=torch.bool)
                for b in range(B):
                    for q in range(T):
                        if mode=='window': break
                        bm[b,q]=cls_of(int(bb[b,q]))[mode]
                if mode!='window': blocked[mode]+=int(bm.sum())
                bmd=bm.to(DEV)
                def fh(mo_,args,o_,mode=mode,bmd=bmd):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=r2(at.c_q),r2(at.c_k)
                    q2,k2=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    ar=torch.arange(T,device=DEV)
                    far=((ar[:,None]-ar[None,:])>=K).float()
                    pat=(sc*sc2)
                    mk=tril.expand(B,T,T).clone()
                    if mode=='window':
                        mk=mk*((ar[:,None]-ar[None,:])<K).float()
                    else:
                        mk=mk*(1-bmd[:,None,:].float()*far[None])
                    z=torch.einsum('bhqk,bkhd->bhqd',pat*tril,
                                   vm.float())
                    z[:,HD]=torch.einsum('bqk,bkd->bqd',
                                         pat[:,HD]*mk,
                                         vm[:,:,HD].float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            for b in range(B):
                toks=fresh[rid[b],:T].tolist(); last={}
                mk2=torch.zeros(T,dtype=torch.bool)
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk2[q]=True
                    last[t]=q
                tot+=float(ce[b][mk2].sum()); n+=int(mk2.sum())
            for h in hs: h.remove()
        return tot/max(n,1)
    base=run(None); win=run('window')-base
    res={}
    for k in KINDS:
        d=run(k)-base
        res[k]={'dce':round(d,5),'blocked':blocked[k],
                'per_1k_blocked':round(1000*d/max(blocked[k],1),5),
                'share_of_window':round(d/max(win,1e-9),3)}
        print(f"{k}: dCE {d:+.5f} blocked {blocked[k]} "
              f"potency {res[k]['per_1k_blocked']:+.5f}/1k "
              f"share {res[k]['share_of_window']}",flush=True)
    pot={k:res[k]['per_1k_blocked'] for k in KINDS}
    top=max(pot,key=pot.get); low=min(pot,key=pot.get)
    rho=spearman([ENRICH[k] for k in KINDS],
                 [pot[k] for k in KINDS])
    pa=(top=='punct'); pb=(low=='subword'); pc=rho>=0.5
    out={'window_dce':round(win,5),'classes':res,
         'most_potent':top,'least_potent':low,
         'spearman_enrichment_vs_potency':round(rho,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'window {win:+.5f} | most potent {top} | least {low} '
          f'| rho(enrichment, potency) {rho:.3f}')
    for nm,v in (('a','punctuation is most potent'),
                 ('b','prose content is least'),
                 ('c','potency tracks enrichment (rho>=0.5)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
