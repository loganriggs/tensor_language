"""Frozen native radial/quadratic split; no fit or discarded harmonic remainder."""
from pathlib import Path
import json,torch
from quartic_harmonic_v1 import lower,evaluate_lower
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'QUARTIC_HARMONIC_NATIVE_V1.json';assert not out.exists()
 r=json.loads((P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_RESULT.json').read_text());assert r['pred_a']
 ap=P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_PROGRAMS.pt';assert digest(ap)==r['artifact_sha256'];p=torch.load(ap,weights_only=True)['programs'][0]
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double();old=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports'];den=old['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
 reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
 trace=p['native_target_traces'];matrix,constant=lower(trace);radial,quadratic=evaluate_lower(trace,x)
 wr=radial@p['output_writers'].T/den[:,None];wq=quadratic@p['output_writers'].T/den[:,None];low=wr+wq;high=reference-low
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 identity=rel(wr+wq+high,reference);trerr=float(matrix.diagonal(dim1=-2,dim2=-1).sum(-1).norm()/matrix.norm())
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);delta=(reference[1::2]-reference[::2])[ids];approx=(low[1::2]-low[::2])[ids]
  families.append(dict(family=family,paired_write_change_error=rel(approx,delta),low_change_norm_ratio=float(approx.norm()/delta.norm())))
 result={'pred_a':max(identity,trerr)<=1e-8,'pred_b':rel(low,reference)<=.1,'pred_c':all(f['paired_write_change_error']<=.1 for f in families),'low_degree_write_error':rel(low,reference),'radial_norm_ratio':float(wr.norm()/reference.norm()),'quadratic_norm_ratio':float(wq.norm()/reference.norm()),'harmonic_remainder_norm_ratio':float(high.norm()/reference.norm()),'families':families,'reconstruction_error':identity,'harmonic_quadratic_trace_error':trerr,'scope':'Exactcanonicalweightsplit; normratiosnotvarianceexplained on text, native denominators retained. No rankfit/semanticpromotion.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
