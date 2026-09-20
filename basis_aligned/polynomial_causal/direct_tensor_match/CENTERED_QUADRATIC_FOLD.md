# Exact third-order tensor inside the mean-centered quartic

20September2026,18:48UTC. This is a coordinate expansion, not a change to the native polynomial. Let

$$
f(x)=C\big[(Lh(x))\odot(Rh(x))\big],\qquad h(x)=D[(Ax)\odot(Bx)].
$$

Set $x=\mu+\delta$, $a=A\mu$, $b=B\mu$, and

$$
h_0=D(a\odot b),\qquad
J=D[\operatorname{diag}(a)B+\operatorname{diag}(b)A],\qquad
h_2(\delta)=D[(A\delta)\odot(B\delta)].
$$

Then $h(\mu+\delta)=h_0+J\delta+h_2(\delta)$. Define the readout derivative with respect to$h$:

$$
F=C[\operatorname{diag}(Lh_0)R+\operatorname{diag}(Rh_0)L].
$$

The first three homogeneous centered-input pieces are

$$
f_0=C[(Lh_0)\odot(Rh_0)],\qquad f_1(\delta)=FJ\delta,
$$

$$
f_2(\delta)=FD[(A\delta)\odot(B\delta)]
+C[((LJ)\delta)\odot((RJ)\delta)].
$$

Thus the quadratic piece has an exact joint three-matrix representation

$$
C_2=[FD\;\;C],\qquad A_2=\begin{bmatrix}A\\LJ\end{bmatrix},\qquad
B_2=\begin{bmatrix}B\\RJ\end{bmatrix},
$$

$$
(T_2)_{vij}=\tfrac12\sum_k(C_2)_{vk}\big[(A_2)_{ki}(B_2)_{kj}+(A_2)_{kj}(B_2)_{ki}\big].
$$

This is precisely half the native pure-quartic Hessian at$\mu$, not an independently fitted tensor. Its two terms arise from different cross-layer computations: the outer-layer derivative reading the inner quadratic, and the outer quadratic acting on the inner linear response. Joint decomposition can combine them; compressing them separately could miss cancellations.

For this native path, input/output dimensions are1152 and channel width is4608+4608=9216. Literal dense three-factor storage costs31,850,496scalars, before the center/constant/linear pieces and output frame. The linear map has1,327,104scalars and the constant1152; lower degree alone does not imply a simpler executable model. These are target operators for later decomposition, not adopted compressed circuits.

The queued degree census measures whether truncation after$f_2$ is useful on both empirical panels and a noncentral Gaussian control, using calibration mean only. It retains all cross-energy terms because the degrees are not orthogonal under actual centered inputs. A useful quadratic approximation would make this joint third-order tensor a focused candidate; a failed baseline should demote it.

`centered_quadratic_factors.py` independently checks constant, linear and quadratic coefficients against automatic differentiation of the original two-layer polynomial. `CENTERED_QUADRATIC_FOLD_CHECK_V1.json` records the result. This does not fold RMSNorm, attention, or other residual contributions into the quartic.
