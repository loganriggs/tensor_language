# Equality L5H5 M4 quadratic context gate V1

Query/key typing does not explain the context-dependent MLP4 tensor geometry.
This experiment tests a label-free state-derived gate with no task, domain,
prompt-family, token, or equality-support input.

On the frozen 192 natural documents, compute the second moment of normalized
MLP4 inputs. Let `P64` project onto its top 64 eigenvectors and freeze

`g(x) = ||P64 x||^2 - E_natural[||P64 x||^2]`.

Partition positions by the sign of this quadratic polynomial gate. Freeze
`P64` and its threshold before examining code. Compute reader-weighted CP-rank
lower bounds and Gaussian-energy audits for both gate arms on natural and code.

Predictions fixed before execution:

1. Each natural arm contains 20--80% of positions; every natural/code arm has
   at least 128 positions, mean state-square in `[.98,1.02]`, PSD error at most
   `2e-4`, and paired unfolding-energy discrepancy at most `2e-4`.
2. Both natural arms have rank-256 lower bound at most `.35`.
3. Each code-arm rank-256 bound differs from its corresponding natural arm by
   at most `.10`.
4. The worse natural arm improves at least 20% on the all-position natural
   lower bound `.4233676211`.
5. Gaussian same-state energy error is at most `.25` in all four cells.

Failure of prediction 1 is invalid only for numerical/state-normalization
failure; an empty or collapsed frozen code arm is a valid nontransfer result.
Other failures are valid gate nulls. Passing licenses a conditional CP fit but
does not itself establish extraction, removal, behavior, or product savings.

Price: one checkpoint load; two natural and one code partial-prefix passes;
four moment/spectral reports; no gradients, fits, parameter updates, or new
text.
