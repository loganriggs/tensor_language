"""Capture a tensor-valued state and its first two derivatives along one ray."""


def second_jet(function, zero):
    import torch
    captured = {}
    direction = torch.ones_like(zero)
    def first_derivative(t):
        value, first = torch.autograd.functional.jvp(
            function, t, direction, create_graph=True)
        captured['value'] = value.detach()
        return first
    with torch.enable_grad():
        first, second = torch.autograd.functional.jvp(
            first_derivative, zero, direction, create_graph=False)
    # Coefficients of 1,t,t², not unscaled derivatives.
    return torch.stack([captured['value'], first.detach(), .5*second.detach()], dim=1)
