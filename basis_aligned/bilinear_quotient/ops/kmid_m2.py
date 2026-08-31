"""LEARNED SECOND-LEVEL INDEX (rung 228): vector-quantized context selection at m2.

CONVENTION (S2135): per-position dCE = CE(m2 replaced, all else real) - CE(real model); LOWER IS BETTER.
S2324: the hand-built 10-class index is too coarse (79% of mass in catch-alls). Here the second level is
LEARNED: k-means (256 centroids, fit on FW m2-inputs, disjoint from census) quantizes the module input;
each cell owns a top-1152 unit row (fit-data mean scores). Runtime selection = nearest centroid -> gather
row: a static selection tensor behind a small vector-quantizer node - still a tensor network, no top-k on
activations.
ARMS (m2; anchors: STATIC 0.2200, TOKID 0.1697, HIER10 0.1634, ORACLE 0.0353):
  KM256  = cluster-indexed rows alone.
  KMHIER = blend s_tok(id) + s_km(cell), top-1152 of the static sum.
REGISTERED PREDICTIONS:
  (a) THE LEARNED INDEX BEATS THE TOKEN INDEX AT DEPTH: census|dCE|(KM256) <= 0.8 x TOKID (<= 0.1358).
  (b) COMPOUNDING: KMHIER <= 0.75 x TOKID (<= 0.1273).
  (c) CONSISTENCY: ORACLE <= KM256 <= STATIC; arms non-inert.
NULL: quantized context cannot select (KM256 >= 0.95 x TOKID) - selection needs fine activation detail
and routing stays dynamic. PRICE: 256x1152 centroids + 256-row table + 2 census passes. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['clsid_m2_results.json','tokid_m2_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: vector-quantized context selection at m2'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'kmid_m2_results.json'
MOD=2; NC=256; K=1152

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    ANC=json.load(open(PT+'tokid_m2_results.json'))['arms']
    mlp=m.transformer.h[MOD].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0)
    W=m.transformer.wte.weight.detach().float(); Vw=W.shape[0]
    STOK=torch.empty(Vw,4608,dtype=torch.float16,device=DEV)
    for v0 in range(0,Vw,4096):
        xe=F.rms_norm(W[v0:v0+4096],(D,))
        STOK[v0:v0+4096]=(((xe@L.T)*(xe@Rw.T)).abs()*dn).half()
    # fit-side capture of m2 inputs + scores
    cap={}
    hp=mlp.register_forward_pre_hook(lambda mo,i_: cap.__setitem__('x',i_[0].detach().float()))
    XS=[]; SS=[]
    for i in range(0,512,4):
        ix=FW[i:i+4,:256].to(DEV)
        with torch.no_grad():
            x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
            for li in range(MOD+1): x,v1=m.transformer.h[li](x,v1,x0)
            xin=cap['x'].reshape(-1,D)
            u=(xin@L.T)*(xin@Rw.T)
            XS.append(xin); SS.append((u.abs()*dn))
    hp.remove()
    X=torch.cat(XS); S=torch.cat(SS)
    del XS,SS
    g=torch.Generator(device='cpu').manual_seed(29)
    CTR=X[torch.randperm(X.shape[0],generator=g)[:NC]].clone()
    for it in range(15):
        d2=(X.pow(2).sum(1,keepdim=True)-2*X@CTR.T+CTR.pow(2).sum(1))
        asg=d2.argmin(1)
        for c in range(NC):
            mkc=asg==c
            if mkc.any(): CTR[c]=X[mkc].mean(0)
    SC=torch.zeros(NC,4608,device=DEV); CC=torch.zeros(NC,device=DEV)
    SC.index_add_(0,asg,S); CC.index_add_(0,asg,torch.ones_like(asg,dtype=torch.float))
    SKM=SC/CC.clamp(min=1)[:,None]
    KIDX=SKM.topk(K,dim=-1).indices
    occ=int((CC>0).sum())
    print(f'k-means fit: {occ}/{NC} cells occupied ({time.time()-T00:.0f}s)',flush=True)
    del X,S,SC,d2
    CN2=CTR.pow(2).sum(1)
    CUR={}
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            b=ROWS[i:i+4,:257].to(DEV)
            ix=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            CUR['ids']=ix
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    def cells(x):
        xf=x.reshape(-1,D)
        return (xf.pow(2).sum(1,keepdim=True)-2*xf@CTR.T+CN2).argmin(1).reshape(x.shape[0],x.shape[1])
    def hk_km(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=KIDX[cells(x)]
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    def hk_hier(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        sc=STOK[CUR['ids']].float()+SKM[cells(x)]
        idx=sc.topk(K,dim=-1).indices
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    res={}
    for nm,hk in (('KM256',hk_km),('KMHIER',hk_hier)):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} inert')
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f}",flush=True)
    pa=res['KM256']['aggabs']<=0.8*ANC['TOKID']['aggabs']
    pb=res['KMHIER']['aggabs']<=0.75*ANC['TOKID']['aggabs']
    pc=(ANC['ORACLE']['aggabs']<=res['KM256']['aggabs']<=ANC['STATIC']['aggabs'])
    out={'arms':res,'anchors':{k:v['aggabs'] for k,v in ANC.items()},'cells_occupied':occ,
         'pred_a_learned_beats_token':bool(pa),'pred_b_compound':bool(pb),'pred_c_consistency':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) KM256 {res['KM256']['aggabs']:.4f} <= 0.8 x TOKID {ANC['TOKID']['aggabs']:.4f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) KMHIER {res['KMHIER']['aggabs']:.4f} <= 0.75 x TOKID: {'HELD' if pb else 'FAILED'}")
    print(f"(c) ORACLE <= KM256 <= STATIC: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
