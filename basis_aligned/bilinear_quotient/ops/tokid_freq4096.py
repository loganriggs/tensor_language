"""TABLE COMPRESSION (rung 233): 4096 frequency-selected rows instead of the full 50k-row m0 table.

CONVENTION (S2135): per-position dCE = CE(m0 replaced, all else real) - CE(real model); LOWER IS BETTER.
S2321's naive table is 58M indices. Zipf says a few thousand rows carry the mass: keep embedding-derived
rows ONLY for the 4096 most frequent tokens (FW counts, disjoint from census); every rare token falls
back to the single static top-1152 row. Price: 4.7M indices - 12x smaller.
ARMS: TOK4096 = this run (anchors from the S2321 receipt: TOKID-full 0.1478, STATIC 0.8019).
REGISTERED PREDICTIONS:
  (a) THE MASS IS IN FEW ROWS: census|dCE|(TOK4096) <= 1.3 x TOKID-full.
  (b) STILL FAR BETTER THAN STATIC: TOK4096 <= 0.5 x STATIC.
  (c) INSTRUMENT: frequent-token census coverage >= 0.90; result differs from both receipts.
NULL: the tail rows matter (TOK4096 >= 1.6 x TOKID-full) - table compression must be smarter than
frequency truncation. PRICE: 4.7M indices + 1 census pass. Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['cp_topk1152_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: frequency-truncated m0 table'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'tokid_freq4096_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    mlp=m.transformer.h[0].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0); K=1152
    # token-indexed table from pure embeddings (wte may be padded past V)
    W=m.transformer.wte.weight.detach().float()
    Vw=W.shape[0]
    IDX=torch.empty(Vw,K,dtype=torch.int32,device=DEV)
    for v0 in range(0,Vw,4096):
        xe=F.rms_norm(W[v0:v0+4096],(D,))
        ue=(xe@L.T)*(xe@Rw.T)
        IDX[v0:v0+4096]=(ue.abs()*dn).topk(K,dim=-1).indices.int()
        del xe,ue
    print(f'token table built ({time.time()-T00:.0f}s)',flush=True)
    cnt=torch.bincount(FW[:512,:256].reshape(-1).to(DEV),minlength=Vw)
    keep=cnt.topk(4096).indices
    kmask=torch.zeros(Vw,dtype=torch.bool,device=DEV); kmask[keep]=True
    _kp=( (Dw.norm(dim=0))*L.norm(dim=1)*Rw.norm(dim=1)).argsort(descending=True)[:K].int()
    IDX[~kmask]=_kp
    covf=float(kmask[ROWS[:,:256].reshape(-1).to(DEV)].float().mean())
    print(f'frequency-truncated to 4096 rows; census coverage {covf:.4f}',flush=True)
    kp=(dn*L.norm(dim=1)*Rw.norm(dim=1)).argsort(descending=True)[:K]
    smask=torch.zeros(4608,device=DEV); smask[kp]=1.0
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
    def hk_static(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        return (((u*smask)@Dw.T)+db).to(o_.dtype)
    def hk_oracle(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=(u.abs()*dn).topk(K,dim=-1).indices
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    def hk_tokid(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=IDX[CUR['ids']].long()
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    ANC=json.load(open(PT+'tokid_sparsity_results.json'))['arms']
    res={}
    for nm,hk in (('TOKID',hk_tokid),):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} inert')
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f}",flush=True)
    ra=res['TOKID']['aggabs']/max(ANC['TOKID']['aggabs'],1e-9)
    rb=res['TOKID']['aggabs']/max(ANC['STATIC']['aggabs'],1e-9)
    pa=ra<=1.3; pb=rb<=0.5
    pc=covf>=0.90 and abs(res['TOKID']['aggabs']-ANC['TOKID']['aggabs'])>1e-4 and abs(res['TOKID']['aggabs']-ANC['STATIC']['aggabs'])>1e-4
    out={'arms':res,'ratio_tokid_static':round(ra,4),'ratio_tokid_oracle':round(rb,4),
         'pred_a_token_carries':bool(pa),'pred_b_near_oracle':bool(pb),'pred_c_consistency':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'ratios: TOKID/STATIC {ra:.3f} | TOKID/ORACLE {rb:.3f}')
    print(f"(a) TOK4096 <= 1.3 x full-table TOKID ({ra:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) TOK4096 <= 0.5 x STATIC ({rb:.3f}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) coverage {covf:.3f} >= 0.90 + non-degenerate: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
