# Subject-number response-weighted prototype freeze V1

Registered before execution and before any fourth-corpus prompts are authored.
This run performs no generalization test. It converts the valid opened-authority
V2 discovery rule into the immutable artifact needed for a prospective test.

Use all 32 opened rows and all 16 E/A/U/W backgrounds. Separately for
`singular_to_plural` and `plural_to_singular`, repeat exactly the valid V2 rule:

1. average the base grouped-MLP6/7 displacement to obtain $p_0$;
2. retain the top 16 right singular vectors $V$ of all background-conditioned
   displacements centered on $p_0$;
3. evaluate $g_i=\nabla_x u^TH_i(x)$ at $x_i+p_0$;
4. fit $s-s_0$ from $g_i^TV$ with ridge
   $\lambda=\operatorname{tr}(A^TA)/16$; and
5. freeze the single vector $p=p_0+Vc$.

The artifact contains exactly two 1,152-vectors, the native L11H3 axis, the
all-row interaction coefficient vector already recorded by the discovery run,
and hashes of every input. It must contain 512 finite gradient rows, preserve
the `<=2` prototype/mean norm-ratio gate, and reproduce a direct reevaluation
of its own vectors bit-for-bit within the run. No answer logits, downstream
behavioral outcomes, new text, rank/ridge sweep, or parameter update is allowed.

Training-panel response and coefficient metrics are descriptive only. The next
runner must bind this artifact before defining its prospective authority. That
runner must not refit vectors or coefficients.
