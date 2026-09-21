"""Opened-state diagnostic: exact substitutions in each two-read component."""
from pathlib import Path
import json,torch
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
graphs=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)
ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();rows=[]
for geometry in ('calibration_shaped','native_isotropic'):
 bundle=expand(graphs[geometry+'_inherited'])
 for j,pair in enumerate(d['pairs']):
  Q=torch.stack(pair['Qs']);g=bundle[str(j)];H=decode(g)
  exact=torch.einsum('ni,oij,nj->no',z,Q,z)
  approx=torch.einsum('ni,oij,nj->no',z,H,z)+z@torch.stack([g['a_linear'],g['b_linear']],1)+torch.stack([g['a_bias'],g['b_bias']])
  def component(reads):return ((h@pair['a']-.5*reads[:,0])/s-pair['alpha'])*(reads[:,1]/s-pair['beta'])
  truth=component(exact);den=(pair['truth'][ids]-pair['truth'][ids].mean()).norm()
  errors={};deltas={}
  for name,mask in [('approximate',(False,False)),('exact_first',(True,False)),('exact_second',(False,True)),('exact_both',(True,True))]:
   reads=torch.stack([exact[:,k] if mask[k] else approx[:,k] for k in range(2)],1)
   delta=component(reads)-truth;errors[name]=float(delta.norm()/den);deltas[name]=delta
  interaction=deltas['approximate']-deltas['exact_first']-deltas['exact_second']
  # Independent identity: product of the two source-read errors with coefficient -1/(2s^2).
  formula=-.5*(approx[:,0]-exact[:,0])*(approx[:,1]-exact[:,1])/s.square()
  replay=float((interaction-formula).norm()/den);assert replay<1e-10
  rows.append(dict(geometry=geometry,pair=j,errors=errors,interaction_error=float(interaction.norm()/den),interaction_identity_replay=replay,cached_truth_replay=float((truth-pair['truth'][ids]).norm()/den)))
out=dict(records=rows,scope='448 previously examined states; h and normalization held native. Exact source-read substitutions diagnose approximation error, not semantic ablations or fresh causal evidence. Error normalized by centered cached truth norm; exact-both uses recomputed native RMS to avoid conflating its small cached-scale drift.')
(P/'SOURCE_READ_REPLACEMENTS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
