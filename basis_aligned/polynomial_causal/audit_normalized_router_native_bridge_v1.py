"""Post-result independent native-helper bridge; no full-model/semantic claim."""
import json
from datetime import datetime, timezone
import torch
from jacclust.tt_model import Rotary, apply_rotary_emb
import folded_normalized_router_v1 as f
from audit_normalized_router_fold_v1 import ROOT, CHECKPOINT, EXPECTED, digest, error


def main():
    out = ROOT / 'NORMALIZED_ROUTER_NATIVE_BRIDGE_V1_RESULT.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    assert digest(CHECKPOINT) == EXPECTED
    state = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    weights = [state[f'transformer.h.0.attn.{n}.weight'][:128].float()
               for n in ['c_q','c_k','c_q2','c_k2']]
    generator = torch.Generator().manual_seed(9111260)
    x,y = [torch.randn(64,1152,generator=generator,dtype=torch.float64) for _ in range(2)]
    x,y = [(z/z.square().mean(-1,keepdim=True).sqrt()).float() for z in (x,y)]
    native_rot = Rotary(128)
    cos,sin = native_rot(torch.empty(1,128,1,128))
    def rotate(z,pos):
        return apply_rotary_emb(z[:,None,None,:],cos[:,pos:pos+1],sin[:,pos:pos+1])[:,0,0,:]
    def native(t,s):
        scores=[]
        for q,k in (weights[:2],weights[2:]):
            qx=torch.nn.functional.rms_norm(x@q.T,(128,))
            ky=torch.nn.functional.rms_norm(y@k.T,(128,))
            scores.append((rotate(qx,t)*rotate(ky,s)).sum(-1)/128)
        return scores[0]*scores[1]
    cells=[]
    for t,s in [(0,0),(1,0),(31,7),(127,32)]:
        n=native(t,s)
        cells.append({'t':t,'s':s,'native_vs_direct':error(f.direct(weights,x,y,t,s),n)})
    lag_left=f.rotary(31,128).T@f.rotary(7,128)
    lag_right=f.rotary(24,128).T@f.rotary(0,128)
    gram=f.rotary(31,128).T@f.rotary(31,128)
    r={'schema':'normalized_router_native_bridge.v1','checkpoint_sha256':EXPECTED,
       'cells':cells,'native_helper_bridge_pass':all(c['native_vs_direct']['relative_l2']<=1e-5 and c['native_vs_direct']['max_abs']<=1e-5 for c in cells),
       'equal_relative_lag_matrix_discrepancy':error(lag_left,lag_right),
       'equal_relative_lag_router_discrepancy':error(native(31,7),native(24,0)),
       'position31_orthogonality_error':error(gram,torch.eye(128,dtype=torch.float64)),
       'scope':'Post-result instrument audit and descriptive rounding identities; no semantic gates.',
       'hashes':{p.name:digest(p) for p in [ROOT/'audit_normalized_router_native_bridge_v1.py',ROOT/'folded_normalized_router_v1.py',ROOT.parents[1]/'jacclust/tt_model.py']},
       'finished_utc':datetime.now(timezone.utc).isoformat(),'model_forwards':0,'gpu_accessed':False}
    with out.open('x') as stream:json.dump(r,stream,indent=2);stream.write('\n')
    print(json.dumps(r))


if __name__=='__main__':main()
