"""Frozen bank coefficient importance and paired-write error accounting; no fitting."""
from pathlib import Path
import json,torch
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 ap=P/'QUARTIC_FULL_QUADRATIC_BANK_V1_PROGRAM.pt';r=json.loads((P/'QUARTIC_FULL_QUADRATIC_BANK_V1_RESULT.json').read_text());assert digest(ap)==r['artifact_sha256']
 p=torch.load(ap,weights_only=True);ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].double();write=p['native_write'].double()
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
 k=p['coefficient_gram'];c=p['target_cross'];a=torch.cat([p['mixing'],p['full_quadratic_mixing']]);capture=(a*c).sum()
 conditional=a.square().sum(-1)/torch.linalg.inv(k).diagonal()
 independent=a.square().sum(-1)*k.diagonal()
 error=write-ref;families=[]
 for fam in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==fam]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
  difference=ref[2*ids+1]-ref[2*ids];de=error[2*ids+1]-error[2*ids];common=(error[2*ids+1]+error[2*ids])/2
  endpoint=error[ep].norm()/ref[ep].norm();delta=de.norm()/difference.norm()
  identity=(error[ep].square().sum()-(2*common.square().sum()+.5*de.square().sum())).abs()/error[ep].square().sum()
  families.append(dict(family=fam,endpoint_relative_error=float(endpoint),paired_difference_relative_error=float(delta),relative_error_amplification=float(delta/endpoint),difference_to_endpoint_reference_norm=float(difference.norm()/ref[ep].norm()),error_pair_identity_relative=float(identity)))
 out=dict(pred_a=max(f['error_pair_identity_relative'] for f in families)<1e-12,coefficient_capture=float(capture),full_node_conditional_capture_fraction=(conditional[-8:]/capture).tolist(),full_node_independent_energy_fraction=(independent[-8:]/capture).tolist(),families=families,artifact_sha256=digest(ap),scope='Post-hoc diagnostic only. Paired write differences precede native RMS/readout nonlinearities; no causal or OOD claims and no fitted changes.')
 path=P/'QUARTIC_FULL_BANK_DIAGNOSIS_V1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['pred_a']
if __name__=='__main__':main()
