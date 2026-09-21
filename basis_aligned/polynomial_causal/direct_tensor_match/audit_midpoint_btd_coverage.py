from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent
torch.set_num_threads(2)
r=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=r['n'].flatten(0,1).double();m=r['m'].flatten(0,1).double();y=r['y'].flatten(0,1).double();y=y-y.mean(0)
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);d=torch.load(p/'MIDPOINT_NATIVE_BTD_V1.pt',weights_only=True)
q=old['scalar_readers'].double();v=old['reduced_writers'].double();exact=(y@q)@v.T
norm=(y@S).norm();omitted=(y-exact)@S
results={}
for name,e in [('fixed',d['fixed_program']),('learned',d['program'])]:
 h=((n@e['A'].double())*(m@e['B'].double()))@e['readout'].double()-e['offset'].double()
 approx=h@e['reduced_writers'].double().T
 retained=(exact-approx)@S
 full=(y-approx)@S
 results[name]=dict(full_variation_relative_error=float(full.norm()/norm),omitted_output_relative_error=float(omitted.norm()/norm),retained_approximation_relative_error=float(retained.norm()/norm),normalized_cross_term=float(2*(retained*omitted).sum()/norm.square()),squared_error_identity_residual=float(abs(full.square().sum()-omitted.square().sum()-retained.square().sum()-2*(retained*omitted).sum())/norm.square()))
assert max(v['squared_error_identity_residual'] for v in results.values())<1e-10
out=p/'MIDPOINT_BTD_COVERAGE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(results=results,scope='Original calibration only. Full vocabulary-centered folded-output variation before final normalization/softcap; fixed mean retained. Not native intervention error. Four-output omitted variation explicitly included.'),indent=2)+'\n');print(out.read_text())
