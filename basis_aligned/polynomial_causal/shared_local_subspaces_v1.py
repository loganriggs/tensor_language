"""Shared/global plus one private subspace per row, exact conditional updates.

No centering inside this module: the caller retains and charges its chosen mean.
Global and private banks may overlap. Joint code solves use minimum-norm least
squares; orthonormal span projections avoid conditioning-dependent assignment.
"""
import torch


def bank(x, rank):
    # Full right basis permits a requested width even for small/empty groups.
    _, _, vh = torch.linalg.svd(x, full_matrices=x.shape[0] < x.shape[1])
    return vh[:rank]


def encode(x, global_bank, local_banks):
    n = len(x)
    g = len(global_bank)
    scores = []
    bases = []
    for local in local_banks:
        joined = torch.cat((global_bank, local))
        u, s, vh = torch.linalg.svd(joined, full_matrices=False)
        active = s > s.max() * max(joined.shape) * torch.finfo(x.dtype).eps
        q = vh[active]
        scores.append((x @ q.T).square().sum(1))
        bases.append((u, s, vh, active))
    labels = torch.stack(scores, 1).argmax(1)
    ac = x.new_zeros(n, g)
    bc = x.new_zeros(n, len(local_banks[0]))
    for k, (u, s, vh, active) in enumerate(bases):
        ix = labels == k
        coef = ((x[ix] @ vh[active].T) / s[active]) @ u[:, active].T
        ac[ix], bc[ix] = coef[:, :g], coef[:, g:]
    return dict(global_bank=global_bank, local_banks=local_banks,
                global_codes=ac, local_codes=bc, labels=labels)


def parts(state):
    global_part = state['global_codes'] @ state['global_bank']
    private_part = torch.zeros_like(global_part)
    for k, local in enumerate(state['local_banks']):
        ix = state['labels'] == k
        private_part[ix] = state['local_codes'][ix] @ local
    return global_part, private_part


def objective(x, state):
    a, b = parts(state)
    return float((x-a-b).square().sum() / x.square().sum())


def stationarity(x, state):
    """Intrinsic bank gradients for normalized squared loss, codes eliminated.

    Fixed-assignment differential check, not a guarantee over group exchanges.
    Each bank has orthonormal rows; overlaps across banks are permitted.
    """
    a,b=parts(state)
    residual=a+b-x
    energy=x.square().sum()
    def projected(codes,err,p):
        grad=2*codes.T@err/energy
        gp=grad@p.T
        tangent=grad-((gp+gp.T)/2)@p
        return float(tangent.norm())
    global_gradient=projected(state['global_codes'],residual,state['global_bank'])
    local_gradients=[]
    for k,p in enumerate(state['local_banks']):
        ix=state['labels']==k
        local_gradients.append(projected(state['local_codes'][ix],residual[ix],p))
    return dict(global_intrinsic_gradient=global_gradient,local_intrinsic_gradients=local_gradients,
                maximum_intrinsic_gradient=max([global_gradient]+local_gradients))


def initialize(x, g, groups, r, seed):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    p = bank(x, g)
    residual = x - (x @ p.T) @ p
    locals_ = []
    for _ in range(groups):
        # Random residual neighborhoods give different initial orientations;
        # no planted labels or private banks are supplied to the fit.
        rows = torch.randperm(len(x), generator=generator, device=x.device)[:max(2*r, r+2)]
        locals_.append(bank(residual[rows], r))
    return encode(x, p, locals_)


def sweep(x, state):
    _, private = parts(state)
    target = x - private
    p = bank(target, len(state['global_bank']))
    ac = target @ p.T
    global_part = ac @ p
    residual = x - global_part
    locals_ = []
    for k, old in enumerate(state['local_banks']):
        ix = state['labels'] == k
        locals_.append(bank(residual[ix], len(old)) if bool(ix.any()) else old.clone())
    # Refit both sets of codes and choose the best group for each complete row.
    return encode(x, p, locals_)


def fit(x, g, groups, r, seed, max_sweeps=200, tolerance=1e-10):
    state = initialize(x,g,groups,r,seed)
    history = [objective(x,state)]
    for _ in range(max_sweeps):
        updated = sweep(x,state)
        value = objective(x,updated)
        if value > history[-1] + 1e-10:
            raise RuntimeError(f'Conditional update increased loss: {history[-1]} -> {value}')
        state = updated
        history.append(value)
        if value < tolerance:
            break
    return state, history
