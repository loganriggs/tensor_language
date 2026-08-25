"""Complexity pricing (spec section 3, M0 subset). All computed, none self-reported.
- bits_params: policy B=16 bits per float param (precision beyond B is free);
  integer/categorical structures priced at entropy (TODO M1).
- bits_structure: AST node count of the replacement source under a fixed grammar
  (here: ast.walk count x 8 bits as a stable proxy until the grammar is frozen).
- edges: len(manifest.inputs).
Library refs: M2. Cardinality measurement: M1."""
import ast

POLICY_B = 16


def bits_params(constants):
    """constants: list of dicts with 'numel' (or 'shape')."""
    total = 0
    for c in constants:
        n = c.get('numel')
        if n is None:
            n = 1
            for s in c['shape']:
                n *= s
        total += n * POLICY_B
    return total


def bits_structure(source_path):
    tree = ast.parse(open(source_path).read())
    return sum(1 for _ in ast.walk(tree)) * 8


def price(source_path, constants, inputs):
    bp = bits_params(constants)
    bs = bits_structure(source_path)
    return {'bits_params': bp, 'bits_structure': bs,
            'bits_total': bp + bs, 'edges': len(inputs),
            'mbits': round((bp + bs) / 1e6, 2)}
