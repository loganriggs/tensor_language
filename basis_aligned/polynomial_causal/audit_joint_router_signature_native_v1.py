"""Scope repair: self-attention ties query and key; off-position test unchanged."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from folded_normalized_router_v1 import fold,direct,evaluate,EPS
from joint_router_polynomial_gram_v1 import quartic_inner
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    torch.set_num_threads(2);sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    matrices=[];signatures=[];weights=[]
    for h in [2,3]:
        w=[sd[f'transformer.h.17.attn.{key}.weight'][h*128:(h+1)*128].double() for key in ['c_q','c_k','c_q2','c_k2']]
        s=fold(w,8,8);weights.append(w);signatures.append(s)
        matrices.append([(branch[0]+branch[0].T)/2 for branch in s])
    a,b=matrices[0];c,d=matrices[1]
    cosine=float(quartic_inner(a,b,c,d)/(quartic_inner(a,b,a,b)*quartic_inner(c,d,c,d)).sqrt())
    generator=torch.Generator().manual_seed(712091);x=torch.randn(4,32,1152,generator=generator,dtype=torch.float64)[0];x=F.rms_norm(x,(1152,),eps=EPS)
    errors=[]
    for w,s in zip(weights,signatures):
        actual=direct(w,x,x,8,8);predicted=evaluate(s,x,x)[0]
        errors.append(float((actual-predicted).norm()/actual.norm()))
    original=json.loads((P/'JOINT_ROUTER_SIGNATURE_NATIVE_V1_RESULT.json').read_text())
    result=dict(scope_correction='Original source8 row used independent query/key slots and is not the physical tied-input self-attention signature. This audit supplies the tied quartic numerator; source7/0 determinant test is unchanged.',self_numerator_cosine=cosine,self_numerator_best_proportional_squared_error=max(0.,1-cosine*cosine),self_tied_score_replay_relative_errors=errors,instrument_passed=max(errors)<=1e-10,distinct_source_numerator_errors={k:original['numerators_by_source_position'][k]['focus_error'] for k in ['7','0']},distinct_source_counterexample=original['focus_two_source_sine_summary'],redteam=dict(narrow_failure='Global shared joint-signature/proportional normalized routing for head17.2/17.3 at the registered positions and formal continuous probes',structural_negative=False,does_not_reject=['Task-specific natural-input agreement','Shared subterms or partially overlapping input spaces','Alternative cross-head groupings'],normalizer_limit='Coefficient proportionality is sufficient for a common algebraic signature, not necessary for all equivalent rational representations. Continuous counterexamples reject global row proportionality on the formal domain; they do not prove text reachability.'))
    (P/'JOINT_ROUTER_SIGNATURE_NATIVE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['instrument_passed']
if __name__=='__main__':main()
