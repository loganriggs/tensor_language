"""Standalone rank-eight contextual-Hessian node for four frozen MLP9 inputs."""
import torch


def execute(state, program):
    """Return responses with shape ``[4, 4, *state.shape[:-1], 1152]``.

    ``state`` is the sole activation port: the native pre-MLP9 residual `z9`.
    All other tensors are frozen packaged weights or directions.
    """
    original_dtype = state.dtype
    state = state.float()
    left = program["left"].float()
    right = program["right"].float()
    projected_down = program["projected_down"].float()
    basis = program["basis"].float()
    directions = program["directions"].float()
    left_directions = program["left_directions"].float()
    right_directions = program["right_directions"].float()
    epsilon = float(program["epsilon"])

    left_state = state @ left.T
    right_state = state @ right.T
    q0 = (left_state * right_state) @ projected_down.T
    s0 = state.square().mean(-1) + epsilon
    first_order = []
    scale_first = []
    for index in range(directions.shape[0]):
        first_order.append((left_directions[index] * right_state + left_state * right_directions[index]) @ projected_down.T)
        scale_first.append(2 * (state * directions[index]).mean(-1))
    outputs = []
    for first in range(directions.shape[0]):
        row = []
        for second in range(directions.shape[0]):
            mixed_numerator = (
                left_directions[first] * right_directions[second]
                + left_directions[second] * right_directions[first]
            ) @ projected_down.T
            mixed_scale = 2 * (directions[first] * directions[second]).mean()
            coefficient = (
                mixed_numerator / s0[..., None]
                - first_order[first] * scale_first[second][..., None] / s0.square()[..., None]
                - first_order[second] * scale_first[first][..., None] / s0.square()[..., None]
                + q0 * (
                    2 * scale_first[first] * scale_first[second] / s0.pow(3)
                    - mixed_scale / s0.square()
                )[..., None]
            )
            row.append(coefficient @ basis.T)
        outputs.append(torch.stack(row))
    return torch.stack(outputs).to(original_dtype)
