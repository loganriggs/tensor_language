"""CPU controls for HT representative choice and shared quadratic features.

No model, discovery fit, GPU, or causal-circuit claims. Reproduce with
python check_quartic_quotient_controls_v1.py; writes the adjacent JSON receipt.
"""
import itertools
import json
from pathlib import Path

import torch

from quartic_bilinear_quotient import execute, sparse_program_price
from quartic_pair_dag_v1 import compile_pairs, execute as monomial_execute


def main():
    torch.set_num_threads(2)
    torch.manual_seed(616)
    rows = []
    for d in (2, 4, 8):
        bank = torch.eye(d, dtype=torch.float64)[None]
        edges = torch.tensor([[0, 0]])
        writer = torch.ones(1, 1, dtype=torch.float64)
        terms = torch.tensor([(i, i, j, j) for i in range(d) for j in range(i, d)]).T
        coefficients = torch.tensor([[1 if t[0] == t[2] else 2 for t in terms.T]], dtype=torch.float64)
        monomials = compile_pairs(terms)
        x = torch.randn(31, d, dtype=torch.float64)
        expected = x.square().sum(-1, keepdim=True).square()
        compact = execute(bank, edges, writer, x)
        expanded = monomial_execute(monomials, x, torch.ones(terms.shape[1], dtype=x.dtype), coefficients)
        error = float(torch.maximum((compact-expected).norm(), (expanded-expected).norm()) / expected.norm())
        representative = torch.einsum('ij,kl->ijkl', bank[0], bank[0])
        sym = sum(representative.permute(p) for p in itertools.permutations(range(4))) / 24
        prices = sparse_program_price(bank, edges, writer)
        compact_multiplies = prices['shared_input_products'] + prices['root_products']
        rows.append(dict(d=d, relative_execution_error=error,
                         unsymmetric_pair_rank=int(torch.linalg.matrix_rank(representative.reshape(d*d, d*d))),
                         symmetric_pair_rank=int(torch.linalg.matrix_rank(sym.reshape(d*d, d*d))),
                         compact_features=prices['features'], compact_scalar_products=compact_multiplies,
                         fixed_monomial_scalar_products=monomials['scalar_products'],
                         fixed_monomial_pair_optimum_proven=monomials['optimal']))
        assert error < 1e-12
        assert rows[-1]['unsymmetric_pair_rank'] == 1
        assert rows[-1]['symmetric_pair_rank'] == d*(d+1)//2
        assert compact_multiplies < monomials['scalar_products']
        assert monomials['optimal']
    payload = dict(rows=rows, all_checks_passed=True,
                   scope='planted algebra/price controls, no native model discovery or behavioral evidence',
                   interpretation='fixed-monomial pair optimality is weaker than shared quadratic-feature simplicity')
    Path(__file__).with_name('QUARTIC_QUOTIENT_CONTROLS_V1_RESULT.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
