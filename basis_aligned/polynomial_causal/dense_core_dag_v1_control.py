"""Exact, small symbolic counterexample: dense core need not preclude reuse.

No native weights, data, GPU, or approximate structure-discovery claim.
Predictions are registered in AGENT_BOARD.md before execution.
"""
import json
from pathlib import Path
import sympy as sp


def main():
    z = sp.symbols('z0:5')
    # Invertible integer coordinate transform, deliberately dense.
    S = sp.Matrix([[1, 1, 1, 1, 1], [1, 2, 1, 1, 1],
                   [1, 1, 2, 1, 1], [1, 1, 1, 2, 1],
                   [1, 1, 1, 1, 2]])
    a, b, c, d, e = S * sp.Matrix(z)
    u, v = a+b, c+d
    mixing = sp.Matrix([[1, 2], [3, 5]])
    dense = [sp.expand(q) for q in mixing * sp.Matrix([u*v, u*e])]
    recovered = [sp.factor_list(q) for q in dense]
    rebuilt = [const * sp.prod(f**n for f,n in terms) for const,terms in recovered]
    common = sp.gcd(dense[0], dense[1])
    proportional = sp.cancel(common/u).free_symbols == set()
    counts = [len(sp.Poly(q, *z).terms()) for q in dense]
    valid_factors = all(sum(n for _,n in terms)==2 and
                        all(sp.Poly(f,*z).total_degree()==1 for f,_ in terms)
                        for _,terms in recovered)
    result = {
        'pred_a_exact_replay': all(sp.expand(q-r)==0 for q,r in zip(dense,rebuilt)),
        'pred_b_shared_linear_factor': bool(proportional and valid_factors),
        'pred_c_dense_monomials_exceed_product_nodes': all(n>2 for n in counts),
        'input_transform_determinant': str(S.det()),
        'output_transform_determinant': str(mixing.det()),
        'dense_unique_monomials_per_output': counts,
        'dense_unique_monomials_across_outputs': len(set().union(*[set(sp.Poly(q,*z).monoms()) for q in dense])),
        'recovered_expressions': [str(q) for q in rebuilt],
        'shared_linear_node': str(common),
        'scalar_variable_product_nodes': 2,
        'scope': 'Exact known-structured integer toy, symbolic factoring each slice then common-factor detection. No noisy/native optimizer or uniqueness guarantee. Costs exclude linear readers, constants, additions and output adapters; these must be charged separately in native comparisons.'
    }
    Path(__file__).with_name('DENSE_CORE_DAG_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
