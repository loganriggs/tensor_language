"""Positive verification of section 98's assembly claim: if L16's bus coordinates
are computed from the token-local residual stream (not attention, not any single
upstream write), a regression from the stream state entering L16 should predict
them well, and a regression from L16's attention output alone should not.

Held-out design: fit ridge regressions on rows 0-60, evaluate R^2 on rows 300-336.
Features: (i) residual stream entering L16 (1152-d, linear ridge); (ii) L16's
attention output (1152-d, linear ridge); (iii) control: stream entering L2 (early
state -- how much is already determined at the bottom?).

REGISTERED PREDICTIONS: (a) stream-in-16 R^2 >= 0.6 (the bus is a function of
local stream state); (b) attention-only R^2 <= 0.3 (attention is not the carrier);
(c) early-stream control <= half of stream-in-16 (the bus is computed along the
way, not readable at L2)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_bus_attention import fwd16
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_bus_assembly_results.json')

@torch.no_grad()
def collect(rows):
    feats={'in16':[],'attn16':[],'in2':[],'bus':[]}
    hs=[m.transformer.h[16].register_forward_pre_hook(
            lambda mod,inp: feats['in16'].append(inp[0].detach().reshape(-1,D).float()) or None),
        m.transformer.h[2].register_forward_pre_hook(
            lambda mod,inp: feats['in2'].append(inp[0].detach().reshape(-1,D).float()) or None)]
    for i in range(0,len(rows),6):
        b=rows[i:i+6].to(DEV)
        mo16,att=fwd16(b, capture_attn=True)
        feats['bus'].append(mo16)
        feats['attn16'].append(att.reshape(-1,D).float())
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    return {k:torch.cat(v) for k,v in feats.items()}

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y16=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y16-Y16.mean(0)).float(), full_matrices=False)
    BUS=orth(Vh[:8].T)
    tr=collect(FW[0:60,:257]); te=collect(FW[300:336,:257])
    # note: fwd16 rows are :257 inputs -> mo16 on 256 positions; in16/in2 hooks
    # fire on the m(...) call with 256-token inputs -- align lengths
    n=min(tr['bus'].shape[0], tr['in16'].shape[0])
    yt=(tr['bus'][:n]@BUS); ye=(te['bus'][:min(te['bus'].shape[0],te['in16'].shape[0])]@BUS)
    out={}
    for tag in ('in16','attn16','in2'):
        Xt=tr[tag][:n]; ne=ye.shape[0]; Xe=te[tag][:ne]
        Xt=Xt-Xt.mean(0); Xe=Xe-Xe.mean(0)
        ytc=yt-yt.mean(0); yec=ye-ye.mean(0)
        lam=1e-2*float((Xt**2).mean())*Xt.shape[1]/Xt.shape[0]
        W=torch.linalg.solve(Xt.T@Xt/Xt.shape[0]+lam*torch.eye(D,device=DEV),
                             Xt.T@ytc/Xt.shape[0])
        pred=Xe@W
        r2=1-float(((yec-pred)**2).mean()/ (yec**2).mean())
        out[tag]=r2
        print(f'{tag:7s}: held-out R^2 {r2:+.3f}',flush=True)
    pa=out['in16']>=0.6; pb=out['attn16']<=0.3
    pc=out['in2']<=0.5*out['in16']
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"(a) local stream predicts bus (>=0.6): {'HELD' if pa else 'FAILED'}")
    print(f"(b) attention alone weak (<=0.3): {'HELD' if pb else 'FAILED'}")
    print(f"(c) early stream <= half: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
