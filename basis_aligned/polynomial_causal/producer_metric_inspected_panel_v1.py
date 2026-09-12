"""Diagnostic reuse of inspected panel; not clean confirmation of the new method."""
from pathlib import Path
import torch,json
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 binding=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 receipt=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_RESULT.json').read_text());ap=P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt';assert digest(ap)==receipt['artifact_sha256']
 fresh=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_RESULT.json').read_text());cachepath=P/'QUARTIC_OUTER32_FRESH_V1_PORTS.pt';assert digest(cachepath)==fresh['artifact_sha256']
 cache=torch.load(cachepath,weights_only=True);p=torch.load(ap,weights_only=True);state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 l,r,d=[state['transformer.h.16.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')]
 ports=cache['ports'];x=ports['input16'].double();producer=((x@l.T)*(x@r.T))@d.T*p['producer_scale'];den=ports['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
 scalar=torch.stack([((producer@a).square()*mu).sum(-1) for a,mu in zip(p['output_readers'],p['outer_weights'])],1)
 write=scalar@p['output_writers'].T/den[:,None];ref=cache['reference_write']
 rows=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json').read_text())['rows']
 result=score([ref,cache['approximate_write'],write],ports['pre']+ports['native_output'],rows,state['lm_head.weight'].float(),fresh['reference_effects'])
 old=result['reports'][0].copy();old['name']='candidate';regression=old==fresh['reports'][1]
 families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
  families.append(dict(family=family,write_relative_error=float((write[ep]-ref[ep]).norm()/ref[ep].norm())))
 result['pred_a']=result['pred_a'] and regression
 result['pred_d']=all(f['write_relative_error']<=.05 for f in families)
 result.update(dict(initial_report_regression=regression,families=families,program_sha256=digest(ap),scope='New fixed weight-only method on a previously inspected panel. Diagnostic comparison, not new clean generalization evidence.'))
 out=P/'PRODUCER_METRIC_INSPECTED_PANEL_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reference_effects'},indent=2));assert result['pred_a']
if __name__=='__main__':main()
