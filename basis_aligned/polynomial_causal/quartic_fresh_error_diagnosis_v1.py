"""Post-result fresh error accounting and weight-only outer-tail isotropy check."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 binding=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 receipt=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_RESULT.json').read_text());path=P/'QUARTIC_OUTER32_FRESH_V1_PORTS.pt';assert digest(path)==receipt['artifact_sha256']
 cache=torch.load(path,weights_only=True);ports=cache['ports'];h=ports['pre']+ports['native_output'];ref=cache['reference_write'];approx=cache['approximate_write'];error=approx-ref
 rows=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json').read_text())['rows']
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True);u=state['lm_head.weight'].float();data=[]
 for i,row in enumerate(rows):
  readers=u[[row['donor_answer_id'],row['donor_foil_id']]]
  exact=h[2*i]+(ref[2*i+1]-ref[2*i]).float();candidate=h[2*i]+(approx[2*i+1]-approx[2*i]).float()
  states=torch.stack([h[2*i],exact,candidate]);logits=30*torch.tanh(F.linear(F.rms_norm(states,(1152,)),readers)/30);margin=logits[:,0]-logits[:,1]
  target=float(margin[1]-margin[0]);predicted=float(margin[2]-margin[0]);actual_error=predicted-target
  s=exact.double();delta=candidate.double()-s;rho=(s.square().mean()+torch.finfo(torch.float32).eps).sqrt()
  tangent=(delta-s*(s@delta)/(1152*rho.square()))/rho
  z=readers.double()@(s/rho);dz=readers.double()@tangent
  dm=(1-torch.tanh(z/30).square())*dz;linear=float(dm[0]-dm[1])
  data.append(dict(row_id=row['row_id'],family=row['family'],lexeme=row['lexeme'],reference_effect=target,candidate_effect=predicted,error=actual_error,first_order_error=linear))
 families=[]
 for family in sorted({r['family'] for r in rows}):
  local=[r for r in data if r['family']==family];e=torch.tensor([r['error'] for r in local],dtype=torch.float64);lin=torch.tensor([r['first_order_error'] for r in local],dtype=torch.float64);order=e.square().argsort(descending=True)
  families.append(dict(family=family,error_rms=float(e.square().mean().sqrt()),first_order_relative_error=float((lin-e).norm()/e.norm()),largest_row_energy_fraction=float(e[order[:1]].square().sum()/e.square().sum()),largest_four_energy_fraction=float(e[order[:4]].square().sum()/e.square().sum()),largest_rows=[local[j] for j in order[:4].tolist()]))
 p=torch.load(P/'QUARTIC_OUTER32_V1_PROGRAM.pt',weights_only=True);w=p['output_writers'];ud=u.double();uw=ud@w;reader=ud.T@uw-len(ud)*ud.mean(0)[:,None]*uw.mean(0)[None,:]
 l,r,d=[state['transformer.h.17.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')];tails=[]
 for m in range(2):
  raw=l.T@((d.T@reader[:,m])[:,None]*r);matrix=(raw+raw.T)/2;a=p['output_readers'][m];mu=p['outer_weights'][m]
  tail=matrix-(a*mu)@a.T;complement=torch.eye(1152,dtype=torch.float64)-a@a.T
  isotropic=float(tail.trace()/1120);capture=isotropic**2*1120
  tails.append(dict(mode=m,tail_frobenius_energy_fraction=float(tail.square().sum()/matrix.square().sum()),isotropic_complement_capture_fraction=float(capture/tail.square().sum()),isotropic_coefficient=isotropic,tail_retained_space_residual=float((tail@a).norm()/matrix.norm()),isotropic_projection_identity=float(abs(float((tail*complement).sum())-float(tail.trace()))/tail.norm())))
 result=dict(pred_a=max(t['tail_retained_space_residual'] for t in tails)<1e-8,families=families,weight_only_tail= tails,rows=data,scope='Post-hoc diagnostic, no exclusions or retuning. First-order expansion of native final RMS/capped readout; outer-tail isotropy is a separate weight-only structural test.')
 out=P/'QUARTIC_FRESH_ERROR_DIAGNOSIS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['pred_a']
if __name__=='__main__':main()
