"""Three folded city-side readers; input is normalized native MLP7 city state."""
import torch.nn.functional as F

def execute(program, normalized_city):
    hidden=F.linear(normalized_city,program['left'])*F.linear(normalized_city,program['right'])
    return F.linear(hidden,program['folded_down'])+program['folded_bias']
