"""All-vocabulary algebra audit of the 54-port conditional suffix."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from affine_bilinear_suffix_v1 import compile_suffix,execute_suffix,EPS
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
@torch.no_grad()
def main():
    out=P/'REGIONAL_TOKEN_SUFFIX_SCALE_V1_RESULT.json';assert not out.exists();torch.set_num_threads(2);torch.manual_seed(61209)
    prior=json.loads((P/'REGIONAL_TOKEN_PATH_COMPILE_V1_RESULT.json').read_text());assert all(prior[k] for k in ('pred_a','pred_b','pred_c'))
    a=torch.load(P/'REGIONAL_TOKEN_PATH_COMPILE_V1_ARTIFACT.pt',weights_only=True);pre=a['pre'][:2].double();K=a['representative_operators'].flatten(1,3).double();donor=a['source_deltas'][:2].flatten(1).double()
    sd=torch.load(CK,mmap=True,weights_only=True,map_location='cpu');L,R,D=[sd['transformer.h.17.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].double();U=sd['lm_head.weight'].double();program=compile_suffix(pre,K,L,R)
    def reference(delta):
        y=pre+torch.einsum('nf,nfd->nd',delta,K);x=F.rms_norm(y,(1152,),eps=EPS);return y+((x@L.T)*(x@R.T))@D.T+bias
    def read(h):return 30*torch.tanh((F.rms_norm(h,(1152,),eps=EPS)@U.T)/30)
    direction=torch.randn_like(donor)*.1
    edits=[('zero',torch.zeros_like(donor))]+[(str(scale),direction*scale) for scale in (1,10,100,1000)]
    baseline_ref=read(reference(edits[0][1]));baseline_fold=read(execute_suffix(edits[0][1],program,D,bias));cells=[]
    for name,delta in edits:
        ref=reference(delta);fold=execute_suffix(delta,program,D,bias);r=read(ref);f=read(fold);state_error=float((ref-fold).norm()/ref.norm());logit_error=float((r-f).norm()/r.norm());effect_error=0. if name=='zero' else float(((r-baseline_ref)-(f-baseline_fold)).norm()/(r-baseline_ref).norm())
        cells.append(dict(edit=name,relative_state_error=state_error,relative_full_logit_error=logit_error,relative_logit_effect_error=effect_error,reference_effect_norm=float((r-baseline_ref).norm()),absolute_logit_error_norm=float((r-f).norm()),absolute_state_error_norm=float((ref-fold).norm())))
    result=dict(matched_large_edit_pass=cells[-1]['relative_logit_effect_error']<=1e-10,cells=cells,vocabulary_rows=len(U),folded_LR_floats_per_context=program['left'][0].numel()+program['right'][0].numel(),scope='Executed scale diagnostic of the same synthetic direction; original std.1 relative-tolerance miss preserved. Not behavioral evidence or a repaired original verdict.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
