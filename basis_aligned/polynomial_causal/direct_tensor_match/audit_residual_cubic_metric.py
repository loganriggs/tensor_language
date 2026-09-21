"""Exact implicit symmetric coefficient norm for (a.z)(z.Q.z)."""
import json
from pathlib import Path
import torch
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).resolve().parent

def norm2(a,Q):
 return (a.square().sum()*Q.square().sum()+2*(Q@a).square().sum())/3

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);g=torch.Generator().manual_seed(917)
 a=torch.randn(7,generator=g,dtype=torch.float64);Q=torch.randn(7,7,generator=g,dtype=torch.float64);Q=(Q+Q.T)/2
 H=(torch.einsum('i,jk->ijk',a,Q)+torch.einsum('j,ik->ijk',a,Q)+torch.einsum('k,ij->ijk',a,Q))/3
 replay=float(abs(H.square().sum()-norm2(a,Q))/H.square().sum());assert replay<1e-12
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);graph=expand(torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True));cov=torch.load(P/'FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt',weights_only=True);iso=torch.load(P/'FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt',weights_only=True);S=torch.linalg.inv(d['inverse_root']);rows=[]
 for j,pair in enumerate(d['pairs']):
  for geometry,transform in [('native',torch.eye(1152,dtype=torch.float64)),('covariance_shaped',S)]:
   a=transform.T@pair['a'];true=transform.T@pair['Qs'][1]@transform
   for name,bundle in [('graph',graph),('covariance_baseline',cov),('isotropic_baseline',iso)]:
    error=transform.T@(decode(bundle[str(j)])[1]-pair['Qs'][1])@transform
    rows.append(dict(component=j+1,geometry=geometry,candidate=name,quadratic_relative_error=float(error.norm()/true.norm()),cubic_relative_error=float((norm2(a,error)/norm2(a,true)).sqrt()),reader_contracted_relative_error=float((error@a).norm()/(true@a).norm())))
 out=dict(control_explicit_vs_implicit=replay,rows=rows,scope='Homogeneous cubic numerator only; affine read corrections yield lower-degree terms. RMS scales, embedding/attention contributions and bias remain explicit. Coefficient error is not text-functional error. Native execution identity needs separate capture verification.')
 (P/'RESIDUAL_CUBIC_METRIC_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for row in rows:
  if row['component']==3:print(row)
if __name__=='__main__':main()
