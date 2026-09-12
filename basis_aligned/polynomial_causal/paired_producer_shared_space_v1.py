"""Weight-only function-space overlap of frozen paired outer branches."""
from pathlib import Path
import json,torch,time
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);start=time.perf_counter()
 binding=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 ap=P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt';receipt=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_RESULT.json').read_text());assert digest(ap)==receipt['artifact_sha256']
 p=torch.load(ap,weights_only=True);state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 l,r,d=[state['transformer.h.16.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')]
 h=p['producer_scale']**2*(d@atom_gram(l,r)@d.T);a,b=p['output_readers'];cross=a.T@h@b;sv=torch.linalg.svdvals(cross)
 orth=max(float((q.T@h@q-torch.eye(32)).norm()/32**.5) for q in (a,b));frame=torch.cat([a,b],1);joint=frame.T@h@frame;values=torch.linalg.eigvalsh(joint)
 r=dict(pred_a=orth<1e-8 and float(sv.max())<=1+1e-8,H_orthogonality_error=orth,principal_cosines=sv.tolist(),shared_principal_directions_099=int((sv>=.99).sum()),shared_principal_directions_09=int((sv>=.9).sum()),
  mean_subspace_capture=float(sv.square().mean()),isotropic_random_subspace_expected_capture=32/1152,
  individual_best_abs_correlations=cross.abs().max(1).values.tolist(),individual_09_matches=int((cross.abs().max(1).values>=.9).sum()),
  combined_gram_eigenvalues=values.tolist(),cross_gram=cross.tolist(),artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-start,
  scope='Quadratic-function Frobenius metric overlap, not observed activation overlap. Principal vectors permit mixtures within each bank; individual correlations do not. Algebraic reuse evidence only, no circuit labels or dimension-independent null significance.')
 out=P/'PAIRED_PRODUCER_SHARED_SPACE_V1.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('cross_gram','combined_gram_eigenvalues','individual_best_abs_correlations')},indent=2));assert r['pred_a']
if __name__=='__main__':main()
