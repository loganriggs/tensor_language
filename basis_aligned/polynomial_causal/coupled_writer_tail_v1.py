"""Functional attention9-output -> raw attention17-input tail.

Implementation in progress: requires independent native replay before scientific use.
The input is the full residual immediately after attention9's residual addition,
not attention9's isolated output. All RMS calls retain native FP32 epsilon.
"""
import torch

EPS = torch.finfo(torch.float32).eps


def rms(x):
    return x * (x.square().mean(-1, keepdim=True) + EPS).rsqrt()


class Tail:
    def __init__(self, state_dict, dtype=torch.float64):
        self.weights = {
            key: value.to(dtype=dtype)
            for key, value in state_dict.items()
            if any(key.startswith(f'transformer.h.{i}.') for i in range(9, 18))
        }

    def weight(self, layer, name):
        return self.weights[f'transformer.h.{layer}.{name}']

    def linear(self, x, layer, name):
        return x @ self.weight(layer, name + '.weight').T

    def mlp(self, x, layer):
        normalized = rms(x)
        hidden = self.linear(normalized, layer, 'mlp.Left') * self.linear(normalized, layer, 'mlp.Right')
        return self.linear(hidden, layer, 'mlp.Down') + self.weight(layer, 'mlp.Down_bias')

    def attention(self, x, inherited, layer):
        x = rms(x)
        batch, length, width = x.shape
        def heads(name):
            return self.linear(x, layer, 'attn.' + name).reshape(batch, length, 9, 128)
        inv = 1 / (10000 ** (torch.arange(0, 128, 2).float() / 128))
        angles = torch.outer(torch.arange(length).float(), inv)
        cosine = angles.cos().bfloat16().to(x.dtype)[None, :, None, :]
        sine = angles.sin().bfloat16().to(x.dtype)[None, :, None, :]
        def rotate(q):
            q = rms(q)
            a, b = q.chunk(2, dim=-1)
            return torch.cat((a * cosine + b * sine, -a * sine + b * cosine), dim=-1)
        q, k, q2, k2 = (rotate(heads(name)) for name in ('c_q', 'c_k', 'c_q2', 'c_k2'))
        first = torch.einsum('bthd,bshd->bhts', q, k) / 128
        second = torch.einsum('bthd,bshd->bhts', q2, k2) / 128
        pattern = (first * second).tril()
        mixture = self.weight(layer, 'attn.lamb')
        values = (1 - mixture) * heads('c_v') + mixture * inherited.reshape(batch, length, 9, 128)
        output = torch.einsum('bhts,bshd->bthd', pattern, values).reshape(batch, length, width)
        return self.linear(output, layer, 'attn.c_proj')

    def __call__(self, after_attention9, initial, inherited):
        x = after_attention9 + self.mlp(after_attention9, 9)
        for layer in range(10, 17):
            coefficients = self.weight(layer, 'lambdas')
            x = coefficients[0] * x + coefficients[1] * initial
            x = x + self.attention(x, inherited, layer)
            x = x + self.mlp(x, layer)
        coefficients = self.weight(17, 'lambdas')
        return coefficients[0] * x + coefficients[1] * initial
