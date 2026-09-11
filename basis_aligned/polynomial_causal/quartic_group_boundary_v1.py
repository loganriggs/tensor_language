"""Exact input/output projector attribution and full-source lift, no fit.
A attribution<=1e-9. B input>=.75delta/output<=.5delta.
C fullinput frozenoutputspace replicas physical<=.25; changed definition/price explicit.
"""
import json
from pathlib import Path
import torch
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'QUARTIC_GROUP_BOUNDARY_V1.json';ap=p/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt';assert not out.exists() and not ap.exists()
    prior=json.loads((p/'QUARTIC_GROUP_NATIVE_V1.json').read_text());assert prior['pred_a']
    gp=p/'QUARTIC_GROUP_PROGRAM_V1.pt';assert digest(gp)==prior['program_sha256'];programs=torch.load(gp,weights_only=True,map_location='cpu')['programs']
    port=p/'QUARTIC_GROUP_PORTS_V1_PORTS.pt';assert digest(port)==prior['ports_sha256'];x=torch.load(port,weights_only=True,map_location='cpu')['input16'].double()
    binding=json.loads((p/'QUARTIC_GROUP_PORTS_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    old=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports'];den=old['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean)
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double() for layer in (16,17) for name in ('Left','Right','Down')];scale=float(state['transformer.h.17.lambdas'][0])
    def quartic(x):
        a=((x@l0.T)*(x@r0.T))@d0.T*scale
        return ((a@l1.T)*(a@r1.T))@d1.T/den[:,None]
    q=[quartic((x@a['bank'])@a['bank'].T) for a in programs];full=quartic(x)
    def project(y,a):return (y@(metric@a['output_writers']))@a['output_writers'].T
    g00=project(q[0],programs[0]);g01=project(q[0],programs[1]);g10=project(q[1],programs[0]);g11=project(q[1],programs[1])
    delta=g11-g00;inp=((g10-g00)+(g11-g01))/2;output=((g01-g00)+(g11-g10))/2
    err=float((delta-inp-output).norm()/delta.norm());ir=float(inp.norm()/delta.norm());orr=float(output.norm()/delta.norm())
    lifted=[project(full,a) for a in programs];replica=relative(*lifted)
    torch.save(dict(projected=[g00,g11],lifted=lifted,full_quartic=full,input_term=inp,output_term=output),ap)
    result=dict(pred_a=err<=1e-9,pred_b=ir>=.75 and orr<=.5,pred_c=replica<=.25,attribution_error=err,
        projected_replica_relative_rms=relative(g00,g11),input_term_over_delta=ir,output_term_over_delta=orr,
        input_output_term_cosine=float((inp*output).sum()/(inp.norm()*output.norm())),lifted_replica_relative_rms=replica,
        lifted_norm_over_full=[float(z.norm()/full.norm()) for z in lifted],artifact_sha256=digest(ap),source_native_sha256=digest(p/'QUARTIC_GROUP_NATIVE_V1.json'),
        scope='Exact same-source counterfactual attribution. Fullinputlift changescomponentdefinition and retainsnative2MLPweights; no compactimplementation or behavioralpromotion yet.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']


if __name__=='__main__':main()
