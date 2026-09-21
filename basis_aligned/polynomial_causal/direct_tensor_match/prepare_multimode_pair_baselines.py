"""Executable separate pair-bank baselines for cross-component reuse tests."""
import torch,json
from pathlib import Path
# Reuse the exact weight/spectrum and per-component scoring specification.
import screen_multimode_shared_bank as screen
from quadratic_pair_blocks import compile_pair,products
P=Path(__file__).resolve().parent;programs={};records=[]
for index,(pair,gram) in enumerate(zip(screen.pairs,screen.grams)):
 basis=screen.eigenbasis(gram,256);shared=screen.inv@basis;cores=[basis.T@M@basis for M in pair['Ms']]
 compiled=compile_pair(*cores)
 program=dict(shared_reader=shared@compiled['input_transform'],product_indices=compiled['product_indices'],product_weights=compiled['product_weights'],h_reader=pair['a'],residual_writer=torch.linalg.solve(screen.mode['R_U'],screen.mode['writer']),alpha=pair['alpha'],beta=pair['beta'])
 mean=screen.mu@shared;smallcov=shared.T@screen.oldcov@shared
 for key,Q,core in zip(['a','b'],pair['Qs'],cores):
  linear=2*Q@screen.mu;constant=screen.mu@Q@screen.mu+torch.trace(screen.oldcov@Q)
  program[key+'_linear']=linear-2*shared@(core@mean)
  program[key+'_bias']=constant-screen.mu@linear+mean@core@mean-torch.trace(smallcov@core)
 values=products(screen.z@program['shared_reader'],program['product_indices'])@program['product_weights']
 reads=[program[k+'_bias']+screen.z@program[k+'_linear']+values[:,j] for j,k in enumerate(['a','b'])]
 phi=((screen.h@pair['a']-.5*reads[0])/screen.scale-pair['alpha'])*(reads[1]/screen.scale-pair['beta'])
 direct=screen.evaluate(pair,basis)[1];replay=float((phi-direct).norm()/direct.norm());assert replay<1e-8
 programs[str(index)]=program
 records.append(dict(mode=index+1,program_replay=replay,source_products=len(compiled['product_weights']),stored_float_scalars=sum(v.numel() for v in program.values() if v.is_floating_point()),stored_integer_indices=program['product_indices'].numel()))
torch.save(programs,P/'MULTIMODE_PAIR_BASELINES_V1.pt')
out=dict(records=records,scope='Three executable independent plain256 pair banks, generated as frozen baselines for sharing their input projections. No joint-sharing or fresh-native claim.')
(P/'MULTIMODE_PAIR_BASELINES_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
