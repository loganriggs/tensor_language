import torch
import factorized_two_qk_attention as factored
import projected_two_qk_attention as expanded
import test_projected_two_qk_attention as native_test


def test_independent_native_reference(monkeypatch):
    # Reuse the independent native implementation, not its expanded compiler.
    monkeypatch.setattr(native_test, 'compile_attention', factored.compile_attention)
    monkeypatch.setattr(native_test, 'execute', factored.execute)
    native_test.test_full_two_qk_value_fold_with_nested_rms_rope_and_causality()


def tensor_values(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(tensor_values(v) for v in obj.values())
    return 0


def test_expanded_equivalence_and_storage():
    torch.manual_seed(679)
    batch, length, width, heads, rank, out = 2, 32, 12, 3, 3, 2
    rand = lambda *shape: torch.randn(*shape, dtype=torch.float64)
    weights = {k: rand(width, width)/3 for k in ('q', 'k', 'q2', 'k2', 'v', 'o')}
    angle = rand(length, width//heads//2)
    args = (weights, rand(batch, length, width), rand(width, rank), rand(width, out),
            rand(batch, length, heads, width//heads), .37, heads,
            angle.cos().bfloat16().double(), angle.sin().bfloat16().double(), .03, .05)
    compact, full = factored.compile_attention(*args), expanded.compile_attention(*args)
    z = rand(batch, length, rank)
    for amplitude in (-2., 0., .5, 1., 3.):
        torch.testing.assert_close(factored.execute(compact, amplitude*z),
                                   expanded.execute(full, amplitude*z), atol=1e-10, rtol=1e-10)
    assert tensor_values(compact) < tensor_values(full)
    print({'factored_tensor_values': tensor_values(compact),
           'expanded_tensor_values': tensor_values(full)})
