"""Readout conditioning, cancellation, and FP32 execution audit of graph fits."""
from pathlib import Path
import torch,json
from shared_mixed_source_graph import component_scalars,source_reads
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);programs=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True);records=[]
def cast(x):
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 return x.float() if isinstance(x,torch.Tensor) and x.is_floating_point() else x
for key,graph in programs.items():
 p=graph['shared_mixed'];L=root@p['left_reader'];R=root@p['right_reader'];W=p['product_weights'].T/d['scales'][:4,None];K=.5*((L.T@L)*(R.T@R)+(L.T@R)*(R.T@L));norms=(K.diag()*W.square().sum(0)).sqrt();raw=torch.einsum('or,ir,jr->oij',W,L,R);tensor=.5*(raw+raw.transpose(-1,-2));ev=torch.linalg.eigvalsh(K);atomenergy=float(norms.square().sum());energy=float(tensor.square().sum())
 f64=component_scalars(d['z'],d['h'],graph);f32=component_scalars(d['z'].float(),d['h'].float(),cast(graph)).double();source64=source_reads(d['z'],graph);source32=source_reads(d['z'].float(),cast(graph)).double()
 scalar=[float((f32[:,j]-f64[:,j]).norm()/(f64[:,j]-f64[:,j].mean()).norm()) for j in range(3)]
 source=[float((source32[:,j]-source64[:,j]).norm()/(source64[:,j]-source64[:,j].mean()).norm()) for j in range(6)]
 records.append(dict(candidate=key,product_gram_condition=float(ev[-1]/ev[0]),gram_min_eigenvalue=float(ev[0]),readout_frobenius_norm=float(W.norm()),sum_atom_norms_over_tensor_norm=float(norms.sum()/tensor.norm()),sum_atom_squared_norms_over_tensor_squared_norm=atomenergy/energy,fp32_scalar_relative_variation_error=scalar,fp32_source_relative_variation_error=source))
 print(records[-1],flush=True)
out=dict(records=records,scope='Cancellation ratios are coefficient-space diagnostics, not proofs of ill-posedness or causal salience. FP32 comparison reuses all32 cached rows and native input boundaries; no new native forward, semantic result or quantization claim.')
(P/'PROFILED_CONDITIONING_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
