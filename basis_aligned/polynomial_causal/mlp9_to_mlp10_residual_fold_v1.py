"""Fixed head9 write -> normalized MLP9 -> block10 residual input changes."""
from directional_mlp_response_context_v1 import prepare, evaluate


def execute(z9, biasfree_mlp9, child_amplitude, remainder_amplitude,
            response_program, reentry_scale):
    context=prepare(z9,biasfree_mlp9,response_program)
    return (reentry_scale*evaluate(child_amplitude,context),
            reentry_scale*evaluate(remainder_amplitude,context))
