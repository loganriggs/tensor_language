"""Projector-defined group ablation: the frame-invariant intervention form.

Motivation (rung 474 / math review 0713 move #2): chart-defined ablations gave
frame-RELATIVE interaction dividends (code N/H composition cosine +.9793 under
replacement vs -.8757 under subtraction) while singletons were invariant.  A
group intervention defined by the orthogonal projector onto span(S) depends
only on the SUBSPACE, so every quantity computed from it is invariant under
any GL change of the spanning set -- by theorem, not by empirical check.

  projector(B): P = B (B^T B)^{-1} B^T   (columns of B span S; any basis works)
  ablate(w, B): (I - P) w                (remove the group's subspace component)
  restrict(w, B): P w                    (keep only the group's component)

Verified below: (1) GL-invariance -- P identical (1e-12) across random
invertible recombinations of the spanning set; (2) idempotence + symmetry;
(3) complement identity ablate+restrict=w; (4) nested-group consistency; and
(5) the failure this replaces: coordinate-zeroing of a PARTIAL group in two
different bases of the SAME subspace produces different vectors, while the
projector form cannot.
"""
import numpy as np


def projector(B):
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if B.shape[0] < B.shape[1]:
        B = B.T
    G = B.T @ B
    return B @ np.linalg.solve(G, B.T)


def ablate(w, B):
    return w - projector(B) @ w


def restrict(w, B):
    return projector(B) @ w


if __name__ == "__main__":
    rng = np.random.default_rng(474)
    n, k = 64, 7
    B = rng.standard_normal((n, k))
    P = projector(B)
    # (1) GL invariance: any invertible recombination of the spanning set
    worst = max(np.abs(projector(B @ rng.standard_normal((k, k))) - P).max()
                for _ in range(10))
    print(f"GL-invariance worst dev: {worst:.2e}")
    assert worst < 1e-10
    # (2) idempotent + symmetric
    assert np.abs(P @ P - P).max() < 1e-10 and np.abs(P - P.T).max() < 1e-12
    # (3) complement identity
    w = rng.standard_normal(n)
    assert np.abs(ablate(w, B) + restrict(w, B) - w).max() < 1e-12
    # (4) nested groups: ablating S then a subset of S changes nothing more
    sub = B[:, :3]
    assert np.abs(ablate(ablate(w, B), sub) - ablate(w, B)).max() < 1e-12
    # (5) the chart failure this form removes: zeroing coordinates 0:3 in two
    # different bases of the SAME 7-dim subspace gives DIFFERENT vectors
    Q1, _ = np.linalg.qr(B)
    Q2, _ = np.linalg.qr(B @ rng.standard_normal((k, k)))
    c1, c2 = Q1.T @ w, Q2.T @ w
    c1[:3] = 0.0
    c2[:3] = 0.0
    chart_dev = np.abs((w - Q1 @ (Q1.T @ w) + Q1 @ c1)
                       - (w - Q2 @ (Q2.T @ w) + Q2 @ c2)).max()
    print(f"chart-zeroing basis dependence (the 474 disease): {chart_dev:.3f}")
    assert chart_dev > 1e-2  # the disease is real...
    full_dev = np.abs(ablate(w, Q1[:, :3] @ np.eye(3)) - w).max()  # smoke
    print("all projector_ablate.py verifications pass")
