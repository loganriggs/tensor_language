"""Model-free conditional attention/MLP response DAG.

Prepared contexts and initial response coordinates are required inputs. Their
native generators remain external dependencies; this is not token-input closure.
"""
import torch
from projected_two_qk_attention import execute as attention
from shared_response_runtime import step
from projected_bilinear_response import readout_prepared


def execute(runtime, attention_programs, initial, contexts, final_context, positions):
    if len(runtime['blocks'])!=len(attention_programs) or len(contexts)!=len(attention_programs):
        raise ValueError('one attention and MLP context per stage required')
    z=initial
    for block,ap,context in zip(runtime['blocks'],attention_programs,contexts):
        scaled=z*block['scale']
        innovation=attention(ap,scaled)-attention(ap,torch.zeros_like(scaled))
        z=step(block,z,context,innovation)
    selected=z[torch.arange(len(z),device=z.device),positions]
    return readout_prepared(runtime['readout'],selected,final_context),selected


def select_attention_examples(program, indices):
    """Slice only explicit example axes; retain shared position/feature tensors."""
    def norm(n):return dict(n,base=n['base'][indices],linear=n['linear'][indices])
    def score(s):return dict(s,base=s['base'][indices],query=s['query'][indices],
                             key=s['key'][indices],qnorm=norm(s['qnorm']),knorm=norm(s['knorm']))
    return dict(program,score1=score(program['score1']),score2=score(program['score2']),
                global_norm=norm(program['global_norm']),value_base=program['value_base'][indices],
                cached_value=program['cached_value'][indices])
