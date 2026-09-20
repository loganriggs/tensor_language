"""Separate fourth-moment correction energy from residual alignment through degree six."""
import json
from pathlib import Path
import torch
from gaussian_quartic_mean import quadratic_moments
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 source=torch.load(P/'QUADRATIC_SKIP_PROGRAM_V1.pt',weights_only=True);s={k:v.double() for k,v in source['programs'][2].items()};base={k:v.double() for k,v in torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4].items()};U=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].double();c=source['primitive_mean'].double();W=s['skip_writer']@s['skip_reader'];w=U.T@W;matched=json.loads((P/'SKIP_MOMENT_MATCH_V1.json').read_text())['records'];rows=[]
 for panel in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  x=panel['rows'].double();y=panel['targets'].double();mu=x.mean(0);center=x-mu;M=center.T@center/len(x);mp,G=quadratic_moments(base['A'],base['B'],mu,M);dm=w@(mp-c);gaussenergy=((w@G)*w).sum(1)+dm.square()
  old=quartic(base,x);error=(y-old)@U;p=(x@base['A'].T)*(x@base['B'].T);delta=(p-c)@w.T;empenergy=delta.square().mean(0);empalignment=(error*delta).mean(0);gaussgain=torch.tensor(next(r for r in matched if r['context']==panel['context'])['matched_gaussian_gain'],dtype=torch.float64);gaussalignment=(gaussgain+gaussenergy)/2
  replay=float((((quartic(s,x)-old)@U)-delta).norm()/delta.norm());assert replay<1e-5
  # gain_G - gain_data = 2*(alignment_G-alignment_data) - (energy_G-energy_data)
  gap=gaussgain-(2*empalignment-empenergy);aligngap=2*(gaussalignment-empalignment);energygap=empenergy-gaussenergy;identity=float((gap-aligngap-energygap).abs().max());assert identity<1e-10
  rows.append(dict(context=panel['context'],gaussian_correction_energy=gaussenergy.tolist(),actual_correction_energy=empenergy.tolist(),gaussian_residual_alignment=gaussalignment.tolist(),actual_residual_alignment=empalignment.tolist(),gain_gap_from_alignment=aligngap.tolist(),gain_gap_from_energy=energygap.tolist(),archive_centering_replay=replay,identity_error=identity))
 result=dict(records=rows,scope='Exact decomposition of matched-Gaussian gain mismatch for fixed quadratic skip. Correction energy uses moments through degree4; residual alignment through degree6. Residual includes constant, so alignment is not exclusively sixth order. No fitting.')
 (P/'SKIP_GAIN_COMPONENTS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
