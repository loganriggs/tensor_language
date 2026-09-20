"""Check extracted API against full-vector projection and common-background composition."""
import json
from pathlib import Path
import torch
from scalar_interventions import remove_modes
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(260936)
 raw=torch.load(P/'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt',weights_only=True);artifact={'program':{k:t.double() for k,t in raw['program'].items()},'residual_writer':raw['residual_writer'].double()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);s={k:t.double() for k,t in torch.load(P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt',weights_only=True)['programs'][.5].items()};x=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'][0]['rows'][:16].double().reshape(2,8,1152);h=torch.randn_like(x)*2;state=torch.randn_like(x);g=torch.tensor([.5,-.3,0.,1.],dtype=x.dtype);g1=g.clone();g1[2:]=0;g2=g-g1
 got=remove_modes(state,h,x,artifact,g);feature=((quartic(s,x.flatten(0,1))-view['output_mean'].double())@view['output_directions'].double()).reshape(2,8,4);den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps;expected=state-(feature*g)@artifact['residual_writer'].T/den
 composed=remove_modes(remove_modes(state,h,x,artifact,g1),h,x,artifact,g2)
 error=float((got-expected).norm()/(expected-state).norm());composition=float((composed-got).norm()/(got-state).norm());noop=float((remove_modes(state,h,x,artifact,torch.zeros_like(g))-state).abs().max());assert error<1e-5 and composition<1e-12 and noop==0
 result=dict(full_vector_reference_relative_edit_error=error,common_background_composition_error=composition,zero_strength_max_error=noop,scope='Actual cached feature inputs with synthetic supplied residual backgrounds. API agrees with full-vector projection; edits compose at residual level only. Does not imply additive CE or semantic selectivity.')
 (P/'SCALAR_INTERFACE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
