"""Fixed observable closure under record permutations; exact stored-dyadic rank witness."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import summary_record_transport_reference as S

BASE = Path(__file__).resolve().parent
OUT = BASE / 'SUMMARY_PERMUTATION_CLOSURE_V1.json'
PRIME = 2147483647


def determinant_mod(matrix):
    rows = []
    for row in matrix.tolist():
        values = []
        for x in row:
            numerator, denominator = float(x).as_integer_ratio()
            values.append(numerator * pow(denominator, -1, PRIME) % PRIME)
        rows.append(values)
    determinant = 1
    for j in range(len(rows)):
        pivot = next((i for i in range(j, len(rows)) if rows[i][j]), None)
        if pivot is None:
            return 0
        if pivot != j:
            rows[j], rows[pivot] = rows[pivot], rows[j]
            determinant = -determinant
        value = rows[j][j]
        determinant = determinant * value % PRIME
        inverse = pow(value, -1, PRIME)
        for i in range(j + 1, len(rows)):
            factor = rows[i][j] * inverse % PRIME
            for k in range(j + 1, len(rows)):
                rows[i][k] = (rows[i][k] - factor * rows[j][k]) % PRIME
            rows[i][j] = 0
    return determinant % PRIME


def main():
    signal.alarm(180)
    torch.set_num_threads(2)
    started = time.perf_counter()
    assert not OUT.exists()
    positive = torch.tensor([[.5, 0.], [0., .25]], dtype=torch.float64)
    negative = torch.tensor([[.5, .25], [1., .5]], dtype=torch.float64)
    assert determinant_mod(positive) == pow(8, -1, PRIME)
    assert determinant_mod(negative) == 0
    assert determinant_mod(positive.flip(0)) == -pow(8, -1, PRIME) % PRIME
    package = BASE / 'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest() == 'e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program = E.load_package(torch.load(package, map_location='cpu', weights_only=True))
    with torch.inference_mode():
        a, offset, c = S.operator(program)
        matrix = a.reshape(512, 2, 23, 23).permute(0, 2, 1, 3).reshape(11776, 46)
        singular = torch.linalg.svdvals(matrix)
        residual = matrix.clone()
        indices = []
        for _ in range(46):
            index = int(residual.square().sum(1).argmax())
            indices.append(index)
            direction = residual[index].clone()
            norm = direction.norm()
            if norm == 0:
                break
            direction /= norm
            # Reorthogonalize to make row selection stable; exact witness is independent of selection.
            for _ in range(2):
                residual -= (residual @ direction)[:, None] * direction
        minor = matrix[indices]
        determinant = determinant_mod(minor) if len(indices) == 46 else 0
    result = {
        'scope': 'minimal linear observable closure under all record permutations on all-bijections affine domain',
        'controls_passed': True, 'matrix_shape': list(matrix.shape),
        'singular_values': singular.tolist(),
        'full_rank_stored_dyadic_certificate': determinant != 0,
        'closure_dimension_if_certified': 1058 if determinant else None,
        'prime': PRIME, 'minor_determinant_mod_prime': determinant,
        'selected_rows': indices,
        'minor_float_hex': [[float(x).hex() for x in row] for row in minor.tolist()],
        'numeric_scope': 'Exact rank certificate for stored FP64 coefficients; not an interval certificate of unrounded native weights and operations',
        'native_coefficients_removed': 0,
        'reference_sha256': hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
        'wall_seconds': time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('minor_float_hex', 'selected_rows', 'singular_values')}, indent=2))
    print('singular extrema:', float(singular[0]), float(singular[-1]))


if __name__ == '__main__':
    main()
