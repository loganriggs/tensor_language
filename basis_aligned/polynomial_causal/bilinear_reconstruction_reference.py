"""Small exact-structure references. No checkpoint loads or GPU work on import.

Algebraic recurrence: M += k1 outer k2 outer v; read (q1 outer q2):M / p**2.
Normalization/position/value producers stay outside this identity and remain live.
"""
from __future__ import annotations

import math
import time
import torch
import torch.nn.functional as F

TENSOR_CAP = 256 * 2**20


def check_shape(shape, dtype=torch.float64):
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if size > TENSOR_CAP:
        raise MemoryError(f"analysis allocation {size} exceeds {TENSOR_CAP}")
    return size


def errors(actual, expected):
    a, b = actual.double(), expected.double()
    delta = a - b
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def close(actual, expected, atol=1e-9, rtol=1e-9):
    out = errors(actual, expected)
    out["passed"] = out["finite"] and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return out


def direct_attention(q1, k1, q2, k2, value):
    b, t, h, p = q1.shape
    check_shape((b, h, t, t), q1.dtype)
    s1 = torch.einsum("bthp,bshp->bhts", q1, k1) / p
    s2 = torch.einsum("bthp,bshp->bhts", q2, k2) / p
    mask = torch.ones(t, t, dtype=torch.bool, device=q1.device).tril()
    scores = (s1 * s2).masked_fill(~mask, 0)
    return torch.einsum("bhts,bshv->bthv", scores, value)


def recurrent_step(q1, k1, q2, k2, value, state=None):
    b, h, p = q1.shape
    shape = (b, h, p, q2.shape[-1], value.shape[-1])
    check_shape(shape, q1.dtype)
    if state is None:
        state = q1.new_zeros(shape)
    state = state + torch.einsum("bhi,bhj,bhv->bhijv", k1, k2, value)
    out = torch.einsum("bhi,bhj,bhijv->bhv", q1, q2, state) / (p * q2.shape[-1])
    return out, state


def recurrent_attention(q1, k1, q2, k2, value):
    state, result = None, []
    for t in range(q1.shape[1]):
        out, state = recurrent_step(q1[:, t], k1[:, t], q2[:, t], k2[:, t], value[:, t], state)
        result.append(out)
    return torch.stack(result, 1)


def cached_attention_step(q1, k1, q2, k2, value, cache=None):
    # Actual KV cache baseline. This reference concatenates for clarity, not kernel speed.
    fresh = [k1[:, None], k2[:, None], value[:, None]]
    cache = fresh if cache is None else [torch.cat((old, new), 1) for old, new in zip(cache, fresh)]
    p = q1.shape[-1]
    scores = (torch.einsum("bhi,bshi->bhs", q1, cache[0]) / p
              * torch.einsum("bhj,bshj->bhs", q2, cache[1]) / q2.shape[-1])
    return torch.einsum("bhs,bshv->bhv", scores, cache[2]), cache


def rotary_tt(x, start=0):
    # TT sign convention; FP64 fixture uses FP64 phases, unlike deployed BF16 phase tables.
    p = x.shape[-1]
    phase = torch.outer(torch.arange(start, start+x.shape[1], device=x.device, dtype=x.dtype),
                        10000.0 ** (-torch.arange(0, p, 2, device=x.device, dtype=x.dtype)/p))
    c, s = phase.cos()[None, :, None], phase.sin()[None, :, None]
    left, right = x.chunk(2, -1)
    return torch.cat((left*c + right*s, -left*s + right*c), -1)


def edit_tensor(x, edits, layer, site, donor=None, start=0):
    for edit in edits:
        if edit["layer"] != layer or edit["site"] != site:
            continue
        local = edit["position"] - start
        if not 0 <= local < x.shape[1]:
            continue
        x = x.clone()
        index = (slice(None), local)
        if "head" in edit:
            index += (edit["head"],)
        if edit["kind"] == "zero":
            x[index] = 0
        elif edit["kind"] == "swap":
            source = (slice(None), edit["position"])
            if "head" in edit:
                source += (edit["head"],)
            x[index] = donor[(layer, site)][source]
        else:
            raise ValueError("unknown intervention")
    return x


class Tiny:
    """Two-layer token-to-logit diagnostic, with live mixing, norms and positions."""
    def __init__(self, seed=909, norm=True, rope=True, device="cpu", dtype=torch.float64):
        self.norm, self.rope = norm, rope
        self.h, self.p, self.d, self.hidden, self.vocab = 2, 4, 16, 32, 8
        g = torch.Generator().manual_seed(seed)
        def weight(*shape, scale=0.25):
            check_shape(shape, dtype)
            return (torch.randn(shape, generator=g, dtype=torch.float64)*scale).to(device=device, dtype=dtype)
        self.embedding = weight(self.vocab, self.d, scale=0.5)
        self.output = weight(self.vocab, self.d)
        self.layers = []
        for i in range(2):
            w = {name: weight(self.h*self.p, self.d) for name in ("q1", "k1", "q2", "k2", "v")}
            w.update(o=weight(self.d, self.h*self.p, scale=0.1),
                     left=weight(self.hidden, self.d, scale=0.1),
                     right=weight(self.hidden, self.d, scale=0.1),
                     down=weight(self.d, self.hidden, scale=0.1),
                     bias=weight(self.d, scale=0.01))
            # Independent coefficients, not a complementary interpolation.
            w["residual_a"], w["embedding_b"] = (0.7, 0.4) if i == 0 else (0.8, -0.15)
            w["value_lambda"] = 0.3 if i == 0 else -0.2
            self.layers.append(w)

    def normalize(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=torch.finfo(x.dtype).eps) if self.norm else x

    def block(self, x, e, first_value, layer, backend, edits, donor, trace, start=0, cache=None):
        w = self.layers[layer]
        live = w["residual_a"]*x + w["embedding_b"]*e
        n = self.normalize(live)
        factors = {name: F.linear(n, w[name]).reshape(*n.shape[:2], self.h, self.p)
                   for name in ("q1", "k1", "q2", "k2", "v")}
        if first_value is None:
            first_value = factors["v"]
        factors["v"] = (1-w["value_lambda"])*factors["v"] + w["value_lambda"]*first_value
        for name in ("q1", "k1", "q2", "k2"):
            factors[name] = self.normalize(factors[name])
            if self.rope:
                factors[name] = rotary_tt(factors[name], start)
        for name in factors:
            factors[name] = edit_tensor(factors[name], edits, layer, name, donor, start)
            trace[(layer, name)] = factors[name]
        args = [factors[name] for name in ("q1", "k1", "q2", "k2", "v")]
        if backend in ("direct_step", "recurrent_step"):
            fn = cached_attention_step if backend == "direct_step" else recurrent_step
            head, cache = fn(*(a[:, 0] for a in args), cache)
            head = head[:, None]
        else:
            fn = direct_attention if backend == "direct" else recurrent_attention
            head = fn(*args)
        head = edit_tensor(head, edits, layer, "head", donor, start)
        trace[(layer, "head")] = head
        attention = F.linear(head.flatten(2), w["o"])
        x = live + attention
        n = self.normalize(x)
        mlp = F.linear(F.linear(n, w["left"])*F.linear(n, w["right"]), w["down"], w["bias"])
        mlp = edit_tensor(mlp, edits, layer, "mlp", donor, start)
        x = edit_tensor(x+mlp, edits, layer, "residual", donor, start)
        trace[(layer, "attention")], trace[(layer, "mlp")], trace[(layer, "residual")] = attention, mlp, x
        return x, first_value, cache

    def __call__(self, tokens, backend="direct", edits=(), donor=None, embeddings=None):
        x = self.normalize(self.embedding[tokens] if embeddings is None else embeddings)
        e, first, trace = x, None, {}
        for layer in range(2):
            x, first, _ = self.block(x, e, first, layer, backend, edits, donor, trace)
        logits = 30*torch.tanh(F.linear(self.normalize(x), self.output)/30)
        return logits, trace

    def stream(self, tokens, backend="recurrent_step", edits=(), donor=None):
        caches, logits, pieces = [None, None], [], {}
        for t in range(tokens.shape[1]):
            x = self.normalize(self.embedding[tokens[:, t:t+1]])
            e, first, trace = x, None, {}
            for layer in range(2):
                x, first, caches[layer] = self.block(
                    x, e, first, layer, backend, edits, donor, trace, start=t, cache=caches[layer])
            logits.append(30*torch.tanh(F.linear(self.normalize(x), self.output)/30))
            for key, value in trace.items():
                pieces.setdefault(key, []).append(value)
        return torch.cat(logits, 1), {key: torch.cat(value, 1) for key, value in pieces.items()}


def small_factors(layer, x):
    """Use the existing small checkpoint's producers; it has different RoPE/no QK norm."""
    n = layer.norm(x)
    b, t, _ = n.shape
    out = {name: layer.rotary(getattr(layer, name)(n).view(b, t, layer.n_head, layer.d_head))
           for name in ("q1", "k1", "q2", "k2")}
    out["v"] = layer.v(n).view(b, t, layer.n_head, layer.d_head)
    return out


def small_forward(model, tokens, backend="direct", edits=(), donor=None, capture=False):
    """Full existing DeepModel, retaining all upstream and downstream computations."""
    x, trace = model.embed(tokens), {}
    for li, (kind, layer) in enumerate(zip(model.spec, model.layers)):
        if kind == "mlp":
            x = layer(x)
        else:
            factors = small_factors(layer, x)
            for name in factors:
                factors[name] = edit_tensor(factors[name], edits, li, name, donor)
                if capture:
                    trace[(li, name)] = factors[name]
            fn = direct_attention if backend == "direct" else recurrent_attention
            head = fn(*(factors[name] for name in ("q1", "k1", "q2", "k2", "v")))
            head = edit_tensor(head, edits, li, "head", donor)
            if capture:
                trace[(li, "head")] = head
            update = layer.o(head.flatten(2))
            x = x+update if layer.residual == "add" else torch.lerp(x, update, layer.scale)
        x = edit_tensor(x, edits, li, "residual", donor)
        if capture:
            trace[(li, "residual")] = x
    return model.head(x), trace


def small_decode_step(model, token, position, caches=None):
    """One full-model decoded token, using the checkpoint's own cached RoPE tables."""
    backend, state = caches if caches is not None else ("recurrent_step", {})
    state = dict(state)
    x = model.embed(token[:, None])
    for li, (kind, layer) in enumerate(zip(model.spec, model.layers)):
        if kind == "mlp":
            x = layer(x)
            continue
        n = layer.norm(x)
        c = layer.rotary.cos_cached[:, position:position+1]
        s = layer.rotary.sin_cached[:, position:position+1]
        def project(name):
            z = getattr(layer, name)(n).view(n.shape[0], 1, layer.n_head, layer.d_head)
            left, right = z.chunk(2, -1)
            return z*c + torch.cat((-right, left), -1)*s
        q1, k1, q2, k2 = [project(name)[:, 0] for name in ("q1", "k1", "q2", "k2")]
        value = layer.v(n).view(n.shape[0], layer.n_head, layer.d_head)
        fn = recurrent_step if backend == "recurrent_step" else cached_attention_step
        head, state[li] = fn(q1, k1, q2, k2, value, state.get(li))
        update = layer.o(head.flatten(1))[:, None]
        x = x+update if layer.residual == "add" else torch.lerp(x, update, layer.scale)
    return model.head(x)[:, 0], (backend, state)


def attention_cost(b, layers, heads, t, p, dv=None, bytes_per_scalar=4):
    dv = p if dv is None else dv
    n = b*layers*heads
    return {
        "direct_state_scalars": n*t*(2*p+dv),
        "recurrent_state_scalars": n*p*p*dv,
        "direct_state_bytes": n*t*(2*p+dv)*bytes_per_scalar,
        "recurrent_state_bytes": n*p*p*dv*bytes_per_scalar,
        "direct_decode_macs": n*t*(2*p+dv),
        "recurrent_decode_macs": 2*n*p*p*dv,
        "direct_prefill_triangular_macs": n*t*(t+1)*(2*p+dv)//2,
        "recurrent_prefill_macs": 2*n*t*p*p*dv,
        "scope": "attention contractions and conventional persistent caches only; projections/MLP/vocab excluded",
    }


def benchmark(fn, device="cpu", repetitions=5):
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    for _ in range(2):
        fn()
    sync()
    times = []
    for _ in range(repetitions):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter()-start)
    values = sorted(times)
    return {"median_seconds": values[len(values)//2], "min_seconds": min(times),
            "max_seconds": max(times), "repetitions": repetitions,
            "kernel": "eager PyTorch/einsum; Python scan; no compilation or fusion"}


def exact_fixture(kind):
    """Rational cubic updates and genuine quadratic attention readers, CPU only.

    The Hadamard map changes the accumulator basis AFTER products. It never rotates
    through an elementwise product or RoPE. Four continuous source/query variables.
    """
    import itertools
    import random
    import sympy as s
    rng = random.Random(909)
    cubic = list(itertools.combinations_with_replacement(range(4), 3))
    quadratic = list(itertools.combinations_with_replacement(range(4), 2))
    def matrix(active=4):
        return s.Matrix(4, 4, lambda i, j: rng.choice((-2, -1, 1, 2)) if j < active else 0)
    shared_value = matrix()
    update, reader = s.zeros(128, 20), s.zeros(80, 128)
    router_coefficients = []
    for h in range(2):
        active = 4 if kind == "independent_routers" else 2
        k1, k2 = matrix(active), matrix(active)
        value = shared_value if kind == "independent_routers" else matrix(active)
        if kind == "perturbed" and h == 1:
            k1[0, 2] += s.Rational(1, 2**20)
        q1, q2, out = matrix(), matrix(), matrix()
        for i, j, v in itertools.product(range(4), repeat=3):
            row = h*64 + i*16 + j*4 + v
            for a, b, c in itertools.product(range(4), repeat=3):
                coeff = k1[i, a]*k2[j, b]*value[v, c]
                if coeff:
                    update[row, cubic.index(tuple(sorted((a, b, c))))] += coeff
            for a, b in itertools.product(range(4), repeat=2):
                qi = quadratic.index(tuple(sorted((a, b))))
                for o in range(4):
                    reader[h*40+o*10+qi, row] += q1[i, a]*q2[j, b]*out[o, v]/16
        # Exact key-product maps: shared values cannot make these functions equal.
        router_coefficients.append(s.kronecker_product(k1, k2))
    hadamard = s.Matrix([[1]])
    while hadamard.rows < 128:
        hadamard = hadamard.row_join(hadamard).col_join(hadamard.row_join(-hadamard))
    inverse = hadamard.T/128
    transformed = hadamard*update
    readers = reader*inverse
    head_updates = []
    for h in range(2):
        masked = s.zeros(128, 20)
        masked[h*64:(h+1)*64, :] = update[h*64:(h+1)*64, :]
        head_updates.append(hadamard*masked)
    return dict(update=transformed, readers=readers, head_updates=head_updates,
                cubic=cubic, quadratic=quadratic, hadamard=hadamard,
                router_functions_identical=router_coefficients[0] == router_coefficients[1])


def exact_span(matrix):
    """Return an exact reachable-span certificate, with coordinate-selector encoder."""
    _, columns = matrix.rref()
    basis = matrix[:, list(columns)]
    _, rows = basis.T.rref()
    rows = list(rows)
    lift = basis*basis[rows, :].inv()
    updates = matrix[rows, :]
    if lift*updates != matrix:
        raise ArithmeticError("rational factorization failed")
    return lift, updates, rows, list(columns)


def rational_attention_check():
    """Exact Fraction identity, separate from floating-point tolerance tests."""
    from fractions import Fraction as Q
    import random
    rng = random.Random(19)
    t, p, v = 5, 2, 2
    def vectors(n):
        return [[Q(rng.randint(-3, 3), 4) for _ in range(n)] for _ in range(t)]
    q1, k1, q2, k2, value = vectors(p), vectors(p), vectors(p), vectors(p), vectors(v)
    state = [[[Q(0) for _ in range(v)] for _ in range(p)] for _ in range(p)]
    for target in range(t):
        for i in range(p):
            for j in range(p):
                for o in range(v):
                    state[i][j][o] += k1[target][i]*k2[target][j]*value[target][o]
        for o in range(v):
            direct = sum(sum(q1[target][i]*k1[source][i] for i in range(p))
                         * sum(q2[target][j]*k2[source][j] for j in range(p))
                         * value[source][o] for source in range(target+1))/p**2
            recurrent = sum(q1[target][i]*q2[target][j]*state[i][j][o]
                            for i in range(p) for j in range(p))/p**2
            assert direct == recurrent
    return {"passed": True, "arithmetic": "fractions.Fraction", "positions": t,
            "domain": "rational projected factors; no normalization or positions"}
