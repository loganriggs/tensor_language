#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4608and1152Gram solves,128cachedvalidationvectors.
"""pred_a normal/native execution identities<=1e-8;
pred_b every neuron-span quadratic matrix error<=.1;
pred_c neuron-span frozen-bank write change<=.05.
Eight full quadratics projected onto native MLP16 product and output spaces.
No data fitting; native input/background remain dependencies. 600sec alarm.
"""
import os, sys, json, time, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from coupled_quartic_writer_v1 import features
from quartic_matrixfree_eigen_v2 import SymmetricCoordinates
from quadratic_producer_projection_v1 import atom_gram,correlations,project,matrix
STEM='QUARTIC_BANK_PRODUCER_SPAN_V1'

@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,gram_dimensions=[4608,1152],quadratics=8)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt')
 assert not out.exists() and not ap.exists();signal.alarm(600)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
 torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 source=P/'QUARTIC_FULL_QUADRATIC_BANK_V1_PROGRAM.pt'
 p=torch.load(source,weights_only=True)
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 l,r,d=[state['transformer.h.16.mlp.'+name+'.weight'].double().cuda() for name in ('Left','Right','Down')]
 coords=SymmetricCoordinates(1152,'cuda')
 q=torch.stack([coords.unpack(row.cuda()) for row in p['full_quadratic_packed']])
 g=atom_gram(l,r);c=correlations(l,r,q)
 neuron,normal=project(g,c)
 output,normal_out=project(d@g@d.T,d@c)
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda()
 den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].double().cuda()
 baseline=p['native_write'].cuda();products=(x@l.T)*(x@r.T)
 retained=features(p['input_readers'].cuda(),p['inner_weights'].cuda(),x)@p['mixing'].cuda()
 mixing=p['full_quadratic_mixing'].cuda();writer=p['output_writers'].cuda()
 reports=[];identities=[normal,normal_out];writes=[]
 for name,coeff in [('neuron_products',neuron),('native_output_readers',d.T@output)]:
  projected=torch.stack([matrix(l,r,a) for a in coeff.T])
  errors=(projected-q).flatten(1).norm(dim=1)/q.flatten(1).norm(dim=1)
  values=products@coeff
  direct=torch.stack([((x@a)*x).sum(-1) for a in projected],1)
  identities.append(float((direct-values).norm()/values.norm()))
  residual=q-projected
  orth=correlations(l,r,residual)
  if name=='native_output_readers':orth=d@orth
  source_cross=c if name=='neuron_products' else d@c
  identities.append(float(orth.norm()/source_cross.norm()))
  write=(retained+values.square()@mixing)@writer.T/den[:,None]
  writes.append(write.cpu())
  reports.append(dict(name=name,matrix_relative_errors=errors.tolist(),native_write_change=float((write-baseline).norm()/baseline.norm()),native_reference_error=float((write-ref).norm()/ref.norm())))
 torch.save(dict(neuron_coefficients=neuron.cpu(),output_readers=output.cpu(),native_writes=torch.stack(writes),source_artifact_sha256=digest(source)),ap)
 result={'pred_a':max(identities)<=1e-8,'pred_b':max(reports[0]['matrix_relative_errors'])<=.1,'pred_c':reports[0]['native_write_change']<=.05}
 result.update(dict(reports=reports,identity_errors=identities,normal_residuals=[normal,normal_out],artifact_sha256=digest(ap),source_artifact_sha256=digest(source),
  additional_neuron_coefficients=neuron.numel(),additional_output_readers=output.numel(),shared_product_parent_floats=l.numel()+r.numel(),shared_output_parent_floats=l.numel()+r.numel()+d.numel(),
  wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Weight-only projection of learned quadratic intermediates onto native product/output spans. Native parent weights charged separately; frozen output mix, reused developmental validation. No circuit or OOD claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
