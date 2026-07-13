"""Tier 0.1: theorem verification (spec §2, test 1). Run: python tests/test_closed_form.py"""

import sys
sys.path.insert(0, "/workspace/tensor_language")

import torch

from mechdecomp.closed_form import (achieved_loss, optimal_value, pinv_rows,
                                    rank_r_solution)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # theorem verification: exact-math regime
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_data(d_in=64, d_out=64, N=2000, m_true=10, noise=0.01):
    Q = torch.linalg.qr(torch.randn(d_in, d_in)).Q
    E = Q[:, :m_true]                                     # orthonormal features
    W = torch.randn(d_out, d_in) / d_in ** 0.5
    S = torch.rand(N, m_true).argsort(1)[:, :5]           # up to 5 active
    ksz = torch.randint(1, 6, (N,))
    X = torch.zeros(d_in, N)
    for i in range(N):
        idx = S[i, :ksz[i]]
        alpha = torch.rand(len(idx)) + 0.5
        X[:, i] = (E[:, idx] * alpha).sum(1)
    X = X + noise * torch.randn(d_in, N)                  # A1: full row rank
    return W.to(DEV), X.to(DEV), E.to(DEV)


def test_matches_optimal_value():
    W, X, _ = make_data()
    Y = W @ X
    for r in (1, 4, 10, 32):
        M = rank_r_solution(Y, X, r)
        got = achieved_loss(M, Y, X).item()
        want = optimal_value(Y, X, r).item()
        assert abs(got - want) <= 1e-4 * max(1.0, want), (r, got, want)
        sv = torch.linalg.svdvals(M)
        assert r >= len(sv) or sv[r] / sv[0] < 1e-4, (r, (sv[r] / sv[0]).item())
    print("PASS matches_optimal_value")


def test_beats_adam():
    W, X, _ = make_data(N=1000)
    Y = W @ X
    r = 6
    want = optimal_value(Y, X, r).item()
    best = float("inf")
    for seed in range(5):
        torch.manual_seed(seed)
        A = torch.randn(W.shape[0], r, device=DEV, requires_grad=True)
        B = torch.randn(r, W.shape[1], device=DEV, requires_grad=True)
        opt = torch.optim.Adam([A, B], lr=1e-2)
        for _ in range(3000):
            loss = ((A @ B @ X - Y) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        best = min(best, loss.item())
    assert best >= want - 1e-3 * max(1.0, want), (best, want)
    print(f"PASS beats_adam (closed-form {want:.4f} <= best Adam {best:.4f})")


def test_recovers_WPS():
    # Problem 1 with r = m_true on data spanning exactly the feature subspace
    # (no noise): M* X = W X and M* should equal W P_S on that subspace.
    W, X, E = make_data(noise=0.0, N=4000)
    Y = W @ X
    M = rank_r_solution(Y, X, 10, ridge=1e-8)
    P = E @ E.T
    # compare action on the feature subspace
    err = ((M @ E - W @ E) ** 2).sum().item() / ((W @ E) ** 2).sum().item()
    assert err < 1e-6, err
    print(f"PASS recovers_WPS (relative err on span {err:.2e})")


def test_problem2_edit():
    # counterfactual targets on a subset: closed form still matches optimum
    W, X, _ = make_data()
    Y = W @ X
    Y2 = Y.clone()
    Y2[:, :200] = torch.randn_like(Y2[:, :200])
    for r in (8, 20):
        M = rank_r_solution(Y2, X, r)
        got = achieved_loss(M, Y2, X).item()
        want = optimal_value(Y2, X, r).item()
        assert abs(got - want) <= 1e-4 * max(1.0, want)
    print("PASS problem2_edit")


if __name__ == "__main__":
    test_matches_optimal_value()
    test_beats_adam()
    test_recovers_WPS()
    test_problem2_edit()
    print("Tier 0.1 theorem verification: ALL PASS")
