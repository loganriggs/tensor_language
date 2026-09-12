"""CPU audit of frozen source sectors and source-coordinate redundancy; no fit."""
from pathlib import Path
import json, hashlib, time
import torch
from attention_source_quartic_v1 import unpack

P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(2);tic=time.perf_counter()
    out=P/'ATTENTION16_SOURCE_CANCELLATION_V1_RESULT.json';assert not out.exists()
    path=P/'MATCHED_PARTNER_ATTENTION16_SOURCE_V1_PORTS.pt'
    receipt=json.loads((P/'MATCHED_PARTNER_ATTENTION16_SOURCE_V1_RESULT.json').read_text())
    assert sha(path)==receipt['artifact_sha'] and receipt['pred_a']
    data=torch.load(path,weights_only=True,map_location='cpu')
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows']
    assert sha(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json')==data['rows_sha']
    sectors=data['scalar_sectors'][:,1];delta=sectors[1::2]-sectors[::2]
    reports=[]
    for family in sorted({r['family'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);d=delta[ids]
        full=d.sum(-1).norm();parts=d[:,1:].norm(dim=0);net=d[:,1:].sum(-1).norm()
        reports.append(dict(family=family,sector_delta_norm_over_full=(d.norm(dim=0)/full).tolist(),
                            attention_sum_norm_over_full=float(net/full),
                            attention_sum_of_norms_over_full=float(parts.sum()/full),
                            cancellation_ratio=float(parts.sum()/net.clamp_min(1e-30))))
    # T=[I,O] has an artificial d-dimensional kernel. Its observable isometry
    # E=T^T(TT^T)^(-1/2) yields E^T T^T A T E=(TT^T)^(1/2) A (TT^T)^(1/2).
    # Test a small exact instance, including a deliberately null source change.
    torch.manual_seed(73120);d=7;o=torch.randn(d,d,dtype=torch.float64)
    a=torch.randn(d,d,dtype=torch.float64);a=(a+a.T)/2
    t=torch.cat([torch.eye(d,dtype=torch.float64),o],dim=1)
    vals,vec=torch.linalg.eigh(t@t.T);hs=(vec*vals.sqrt())@vec.T;hi=(vec*vals.rsqrt())@vec.T
    e=t.T@hi;b=t.T@a@t;null=torch.cat([-o,torch.eye(d,dtype=torch.float64)])
    controls=dict(isometry=float((e.T@e-torch.eye(d)).norm()),
                  quotient_identity=float((e.T@b@e-hs@a@hs).norm()/b.norm()),
                  artificial_null=float((b@null).norm()/(b.norm()*null.norm())))
    assert max(controls.values())<1e-12
    result=dict(families=reports,quotient_controls=controls,source_sha=sha(Path(__file__)),
                ports_sha=sha(path),execution_seconds=time.perf_counter()-tic,
                scope='Exact signed scalar sectors on frozen fresh endpoints, not CE attribution or a fitted sparse decomposition.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
