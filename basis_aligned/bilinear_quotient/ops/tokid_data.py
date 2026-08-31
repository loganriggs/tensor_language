"""DATA-DERIVED TOKEN TABLE AT m0 (rung 224): does fit-data statistics close the context residual?

CONVENTION (S2135): per-position dCE = CE(m0 replaced, all else real) - CE(real model); LOWER IS BETTER.
S2321: the embedding-derived token table scores 0.1478 (STATIC 0.8019, ORACLE 0.0349). Here the table row
for each vocab id is the MEAN per-occurrence score over the FIT windows (FW, disjoint from census rows;
leakage-free), with the embedding-derived row as fallback for unseen ids. Still a static selection tensor.
REGISTERED PREDICTIONS (arms: TOKID2 = this run; receipts TOKID 0.1478, ORACLE 0.0349):
  (a) DATA CLOSES MOST OF THE RESIDUAL: census|dCE|(TOKID2) <= 0.6 x TOKID (<= 0.0887).
  (b) NEAR-ORACLE: TOKID2 <= 2.5 x ORACLE (<= 0.0873).
  (c) SANITY: TOKID2 < TOKID; non-inert; fit coverage of census positions >= 0.90.
NULL: typical-context statistics do not transfer per position (TOKID2 >= 0.9 x TOKID) - the residual is
positional, not typical-context. PRICE: table 50257 x 1152 indices (as S2321) + one census pass.
Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['tokid_sparsity_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: data-derived token table at m0'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'tokid_data_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    ANC=json.load(open(PT+'tokid_sparsity_results.json'))['arms']
    mlp=m.transformer.h[0].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0); K=1152
    W=m.transformer.wte.weight.detach().float(); Vw=W.shape[0]
    SUM=torch.zeros(Vw,4608,device=DEV); CNT=torch.zeros(Vw,device=DEV)
    cap={}
    hp=mlp.register_forward_pre_hook(lambda mo,i_: cap.__setitem__('x',i_[0].detach().float()))
    blk=m.transformer.h[0]
    for i in range(0,512,4):
        ix=FW[i:i+4,:256].to(DEV)
        with torch.no_grad():
            x=F.rms_norm(m.transformer.wte(ix),(D,))
            blk(x,None,x)
            u=(cap['x']@L.T)*(cap['x']@Rw.T)
            s=(u.abs()*dn).reshape(-1,4608)
        fid=ix.reshape(-1)
        SUM.index_add_(0,fid,s); CNT.index_add_(0,fid,torch.ones_like(fid,dtype=torch.float))
    hp.remove()
    seen=CNT>0
    MEAN=torch.where(seen[:,None],SUM/CNT.clamp(min=1)[:,None],torch.zeros(1,device=DEV))
    del SUM
    IDX=torch.empty(Vw,K,dtype=torch.int32,device=DEV)
    for v0 in range(0,Vw,4096):
        sc=MEAN[v0:v0+4096].clone()
        us=~seen[v0:v0+4096]
        if us.any():
            xe=F.rms_norm(W[v0:v0+4096][us],(D,))
            sc[us]=((xe@L.T)*(xe@Rw.T)).abs()*dn
        IDX[v0:v0+4096]=sc.topk(K,dim=-1).indices.int()
    cov=float((CNT[ROWS[:,:256].reshape(-1).to(DEV)]>0).float().mean())
    print(f'data table built; seen ids {int(seen.sum())}; census coverage {cov:.4f} ({time.time()-T00:.0f}s)',flush=True)
    CUR={}
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            b=ROWS[i:i+4,:257].to(DEV)
            ix=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            CUR['ids']=ix
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
                for blk2 in m.transformer.h: x,v1=blk2(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    def hk(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=IDX[CUR['ids']].long()
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    h=mlp.register_forward_hook(hk)
    cev=evalce()
    h.remove()
    d=cev-CBASE
    if float(d.abs().max())<1e-6: raise SystemExit('INSTRUMENT FAIL: TOKID2 inert')
    agg=round(float(d.abs().mean()),4)
    print(f'  TOKID2: agg {float(d.mean()):+.4f} | census|dCE| {agg:.4f}',flush=True)
    pa=agg<=0.6*ANC['TOKID']['aggabs']
    pb=agg<=2.5*ANC['ORACLE']['aggabs']
    pc=(agg<ANC['TOKID']['aggabs']) and cov>=0.90
    out={'TOKID2':agg,'anchors':{k:v['aggabs'] for k,v in ANC.items()},'coverage':cov,
         'pred_a_data_closes':bool(pa),'pred_b_near_oracle':bool(pb),'pred_c_sanity':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) TOKID2 {agg:.4f} <= 0.6 x TOKID {ANC['TOKID']['aggabs']:.4f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) TOKID2 {agg:.4f} <= 2.5 x ORACLE {ANC['ORACLE']['aggabs']:.4f}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) improves on TOKID and coverage {cov:.3f} >= 0.90: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
