"""Accept native headed and fixture-flat inherited value layouts identically."""
from coupled_attention10_ports_v1 import compile_program,MAPS,normalized_ports,attention_write


def execute(program,a,b):
    first=program['first_values'];batch,tokens=program['q1'].shape[:2]
    if first.shape not in ((batch,tokens,1152),(batch,tokens,9,128)):
        raise ValueError('First values must be B,T,1152 or B,T,9,128')
    return attention_write(normalized_ports(program,a,b),program['output'],program['mixture'],first.reshape(batch,tokens,1152))
