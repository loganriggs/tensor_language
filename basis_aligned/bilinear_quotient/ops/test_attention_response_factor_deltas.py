#!/usr/bin/env python3
"""Focused algebra and validation tests for exact attention-response factors."""

import torch

import attention_source_destination_eval as subject


def capture(seed):
    generator = torch.Generator().manual_seed(seed)
    batch, length, heads, width = 2, 5, 3, 4
    pattern = torch.randn(batch, heads, length, length, generator=generator)
    value = torch.randn(batch, length, heads, width, generator=generator)
    output = torch.einsum("bhqs,bshd->bqhd", pattern, value)
    return {"pattern": pattern, "value": value, "head_output": output}


def main():
    base, changed = capture(1), capture(2)
    destinations, sources, heads = (3, 4), ((1, 2), (0, 2, 4)), (0, 2)
    factors = subject.attention_response_factor_deltas(
        base, changed, destinations, sources, selected_heads=heads)
    assert tuple(factors) == subject.RESPONSE_FACTORS
    total = sum(factors.values(), torch.zeros_like(next(iter(factors.values()))))
    for index, (destination, row_sources) in enumerate(zip(destinations, sources)):
        for head in heads:
            expected = sum(
                (changed["pattern"][index, head, destination, source]
                 * changed["value"][index, source, head]
                 - base["pattern"][index, head, destination, source]
                 * base["value"][index, source, head])
                for source in row_sources
            )
            assert torch.allclose(total[index, destination, head], expected, atol=2e-6)
    mask = total != 0
    for index, destination in enumerate(destinations):
        mask[index, destination, list(heads)] = False
    assert not bool(mask.any())
    try:
        subject.attention_response_factor_deltas(
            base, changed, destinations, ((), sources[1]), selected_heads=heads)
    except subject.AttentionSourceDestinationError:
        pass
    else:
        raise AssertionError("empty source coverage was accepted")
    print("PASS attention response factor algebra")


if __name__ == "__main__":
    main()
