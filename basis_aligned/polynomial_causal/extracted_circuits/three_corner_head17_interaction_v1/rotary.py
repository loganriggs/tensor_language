import torch

def rotary(position, width):
    """Native half-split convention, including BF16 table rounding."""
    inv = 1.0 / (10000 ** (torch.arange(0, width, 2).float() / width))
    angle = (torch.tensor(float(position), dtype=torch.float32) * inv)
    c, s = angle.cos().bfloat16().double(), angle.sin().bfloat16().double()
    return torch.cat((torch.cat((c.diag(), s.diag()), 1),
                      torch.cat((-s.diag(), c.diag()), 1)), 0)

