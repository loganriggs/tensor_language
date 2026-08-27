# whole_model_heldout: THE WHOLE-MODEL PROGRAM ON DOCUMENTS IT HAS NEVER SEEN
#
# The arc's headline is now a single number -- a 36-site compiled program reproducing 55.04% of
# bilin18's 5.5684-nat joint stake (§1696), with v1 included for free (§1698) and the shortfall
# attributed to cross-half compounding rather than redundancy (§1697). Every one of those
# figures was scored on fineweb_n192_skip7000.
#
# The two halves have each been confirmed held-out separately -- §1683 for the MLP program arms,
# §1693 for the compressibility ordering -- but the JOINT program never has, and it is the
# object all the later conclusions are stated about. §1697's damping factors and §1696's
# transfer discount are both differences between joint-condition ceilings; if those ceilings
# move under a document resample, the discount numbers move with them.
#
# Rung 2, house second-class-confirmation pattern (§1595, §1598, §1603). Programs compiled ONCE
# on n480_skip80 with the mask pinned, then scored on both eval sets. Only the scoring documents
# change.
#
# ARMS, the four the arc's conclusions rest on, all with v1 tabled per §1698:
#   simple            linear everywhere, lag-1        §1694/§1697 baseline
#   attn_upgraded     lags (1,2,4,8)                  §1697's +2.66 arm
#   mlp_upgraded      tables at mlp0-2                §1697's +1.31 arm
#   both              the §1696 program               the headline, 55.04%
#
# The stake is recomputed per eval set because it is a property of the eval documents (§1683
# found the MLP stake moving 4.3301 -> 4.5173 between these two); the ceiling is a ratio within
# its own set, which is what makes them comparable.
#
# Registered predictions:
#   pred_a THE HEADLINE HOLDS: the both arm on skip11000 is within 3 points of 55.04%.
#   pred_b THE GAIN STRUCTURE HOLDS, which is what §1696 and §1697 actually use: on the held-out
#          set all three gains are positive, attention exceeds MLP, the joint gain retains at
#          least half of §1697's 4.10 points, and the singles sum within 1 point of the joint.
#   pred_c CONTROLS: the skip7000 arm reproduces §1696's 55.04% and §1697's simple 50.94%, both
#          within 0.5 points, and the baseline CE reproduces 3.29205 (§1695).
#   pred_d PRECISION: the held-out both-arm document-cluster 95% interval lies wholly inside
#          the registered 55.04% +/- 3-point band; the joint gain and both conditional
#          increments have positive lower bounds; and the interaction interval is inside
#          +/-1 point. This is the uncertainty-aware gate; pred_a is only a point estimate.
import hashlib, json, subprocess, time, sys, os, traceback, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_heldout_results.json'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
EVAL_ROWS = EVAL_SETS[0][1]
CONSTS = PT + 'opt_ablation_consts_all.pt'
m = DEV = H = None
arm_gains = document_cluster_bootstrap = gain_structure_holds = None
S1694_SIMPLE = 0.5094
S1696_BOTH = 0.5504
S1697_GAINS = {'attn': 0.0266, 'mlp': 0.0131, 'both': 0.0410}
S1694_JOINT_STAKE = 5.5684
S1687_ATTN_BEST = 0.6805
S1672_MLP_TABLE_SITES = (0, 1, 2)
ATTN_LAGS = (1, 2, 4, 8)
S1683_CE_LIVE = 3.29205
CFG = {'lags': ATTN_LAGS, 'tables': S1672_MLP_TABLE_SITES, 'v1': None}
V1P = {}
STATE = {}
SEENREF = {}
EXPECTED_INPUT_SHA256 = {
    '.rowcache/fineweb_n480_skip80.pt': '2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496',
    '.rowcache/fineweb_n96_skip80.pt': '94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda',
    '.rowcache/fineweb_n192_skip7000.pt': 'd66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c',
    '.rowcache/fineweb_n192_skip11000.pt': 'b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868',
    '.rowcache/fineweb_oracle_v2_receipt.json': '815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16',
    'opt_ablation_consts_all.pt': '6bba9c4baa753769f740457515a26a358aace8b9ed861cc295dd597aad58da3b',
}
EXPECTED_SHAPES = {
    '.rowcache/fineweb_n480_skip80.pt': (480, 513),
    '.rowcache/fineweb_n96_skip80.pt': (96, 513),
    '.rowcache/fineweb_n192_skip7000.pt': (192, 513),
    '.rowcache/fineweb_n192_skip11000.pt': (192, 513),
}
SOURCE_CLOSURE = (
    'basis_aligned/bilinear_quotient/whole_model_heldout.py',
    'basis_aligned/bilinear_quotient/whole_model_heldout_stats.py',
    'basis_aligned/bilinear_quotient/test_whole_model_heldout_stats.py',
    'basis_aligned/bilinear_quotient/whole_model_heldout_protocol_amendment.json',
    'basis_aligned/bilinear_quotient/bilin18_joint_removal.py',
    'basis_aligned/qk_mdl/tier2_model.py',
    'jacclust/tt_model.py',
)
MODEL_SNAPSHOT = '/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd'
MODEL_REVISION = 'ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'
MODEL_FILES = {
    'config.json': '428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c',
    'pytorch_model.bin': '680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    """Bind the prospective replication to exact rows, constants, and provenance."""
    if os.path.exists(OUT):
        raise FileExistsError(f'create-only result already exists: {OUT}')
    actual = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = PT + relative
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f'input hash mismatch for {relative}: {digest} != {expected}')
        actual[relative] = digest
    for relative, shape in EXPECTED_SHAPES.items():
        raw = torch.load(PT + relative, map_location='cpu')
        raw = raw['rows'] if isinstance(raw, dict) else raw
        if tuple(raw.shape) != shape or raw.dtype != torch.int64:
            raise RuntimeError(f'row contract mismatch for {relative}: {raw.shape} {raw.dtype}')
    receipt = json.load(open(RECEIPT))
    sets = receipt['document_provenance']['sets']
    expected_lengths = {'n480_skip80': 480, 'n96_skip80': 96,
                        'n192_skip7000': 192, 'n192_skip11000': 192}
    for name, length in expected_lengths.items():
        if len(sets[name]) != length:
            raise RuntimeError(f'provenance length mismatch for {name}')
    fit_docs = {e['document_id'] for e in sets['n480_skip80']}
    ref_docs = {e['document_id'] for e in sets['n192_skip7000']}
    held_docs = {e['document_id'] for e in sets['n192_skip11000']}
    if fit_docs & ref_docs or fit_docs & held_docs or ref_docs & held_docs:
        raise RuntimeError('fit/reference/held-out source documents overlap')
    return actual, receipt


def verify_source_binding():
    """Require every outcome-defining source to be a pushed, committed-clean blob."""
    repo = subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=os.path.dirname(__file__), text=True
    ).strip()
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True
    ).strip()
    hashes = {}
    for relative in SOURCE_CLOSURE:
        committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=repo)
        committed_hash = hashlib.sha256(committed).hexdigest()
        current_hash = sha256(os.path.join(repo, relative))
        if committed_hash != current_hash:
            raise RuntimeError(f'source is not the committed blob: {relative}')
        hashes[relative] = current_hash
    pushed = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], cwd=repo
    )
    if pushed.returncode != 0:
        raise RuntimeError(f'executing commit is not on origin/main: {commit}')
    return {'commit': commit, 'hashes': hashes}


def verify_model_binding():
    """Pin the offline Hub revision and exact config/weight bytes before model import."""
    reference = os.path.join(MODEL_SNAPSHOT, 'refs', 'main')
    if open(reference).read().strip() != MODEL_REVISION:
        raise RuntimeError('bilin18 cached Hub revision drifted')
    hashes = {}
    for name, expected in MODEL_FILES.items():
        path = os.path.join(MODEL_SNAPSHOT, 'snapshots', MODEL_REVISION, name)
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f'bilin18 model hash mismatch for {name}')
        hashes[name] = digest
    return {'repo': 'Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd',
            'revision': MODEL_REVISION, 'hashes': hashes}


def write_create_only_atomic(path, payload):
    """Publish a complete result without truncating or replacing an existing one."""
    temporary = f'{path}.tmp.{os.getpid()}'
    try:
        with open(temporary, 'x') as f:
            json.dump(payload, f, indent=1,
                      default=lambda o: sorted(o) if isinstance(o, set) else str(o))
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def lagged(x):
    """[x_t, x_{t-l} for l in CFG['lags']], zero-padded at the start (§1685/§1686)."""
    lags = CFG['lags']
    parts = [x] + [torch.cat([torch.zeros_like(x[:, :l]), x[:, :-l]], dim=1) for l in lags]
    return torch.cat(parts, dim=-1).reshape(-1, D * (1 + len(lags)))


def table_hook(tbl):
    def hook(mod, args, out):
        sub = tbl[STATE['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def mlp_const_hook(c):
    def hook(mod, args, out):
        return c.to(out.dtype).expand_as(out)
    return hook


def attn_const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def identity_hook(mod, args, out):
    """Known-answer intervention exercising the same 36 forward-hook sites."""
    return out


def mlp_prog_hook(W):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def sub_v1(v):
    """Replace the v1 tensor block 0 exports, per CFG['v1']."""
    mode = CFG['v1']
    if mode is None or 'W' not in V1P:
        return v
    # v1 is the HEAD-SPLIT view (B, T, n_head, head_dim) -- last dim is 128, not D.
    # n_head * head_dim == D and the two dims are adjacent, so reshape(-1, D) flattens
    # them correctly. The first build indexed v.shape[-1] and died on 1152 vs 128.
    flat = v.reshape(-1, D)
    if mode == 'table':
        new = V1P['W'][STATE['idx'].reshape(-1)]
    else:
        new = V1P['x'].reshape(-1, D) @ V1P['W']
    new = torch.where(SEENREF['m'][STATE['idx']].reshape(-1).unsqueeze(-1), new, flat)
    return new.reshape(v.shape).to(v.dtype)


def attn_prog_hook(W, L=None):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = (lagged(args[0]) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, y)
        if not isinstance(out, tuple):
            return sub
        rest = list(out[1:])
        if L == 0 and rest and torch.is_tensor(rest[0]):
            V1P['x'] = args[0]
            rest[0] = sub_v1(rest[0])
        return (sub,) + tuple(rest)
    return hook


def install(prog):
    hs = []
    for (kind, L), W in prog.items():
        if kind == 'mlp':
            hs.append(H[L].mlp.register_forward_hook(
                table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W)))
        else:
            hs.append(H[L].attn.register_forward_hook(attn_prog_hook(W, L)))
    return hs


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def fit_table(rows, L, prog):
    """Per-token mean of mlp_L's output with the stack below substituted (§1661 hybrid)."""
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert float(c.sum()) > 0, f'mlp{L}: no token counts'
    sn = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    return tbl


@torch.no_grad()
def fit_site(rows, kind, L, prog):
    if kind == 'mlp' and L in CFG['tables']:
        return fit_table(rows, L, prog)
    din = D if kind == 'mlp' else D * (1 + len(CFG['lags']))
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = (args[0].reshape(-1, D) if kind == 'mlp' else lagged(args[0])).double()
        y = (out if kind == 'mlp' else (out[0] if isinstance(out, tuple) else out))
        A.add_(x.T @ x); B.add_(x.T @ y.reshape(-1, D).double()); n['v'] += x.shape[0]
        return None
    tgt = H[L].mlp if kind == 'mlp' else H[L].attn
    sweep(rows, hooks=install(prog) + [tgt.register_forward_hook(collect)])
    assert n['v'] > 0, f'{kind}{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows, kinds):
    """Interleaved bottom-up: within block L, attn_L then mlp_L (§1669)."""
    prog = {}
    for L in ALL18:
        for kind in ('attn', 'mlp'):
            if kind in kinds:
                prog[(kind, L)] = fit_site(rows, kind, L, prog)
    return prog


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, seen, hooks=(), details=False):
    acc = {'t': 0.0, 'n': 0, 'loss_sums': [], 'counts': []}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
        if details:
            for row_loss, row_cov in zip(e, cov):
                acc['loss_sums'].append(float(row_loss[row_cov].sum()))
                acc['counts'].append(int(row_cov.sum()))
    sweep(rows, hooks=hooks, score=score)
    if acc['n'] <= 0:
        raise RuntimeError('no covered scoring positions')
    if not torch.isfinite(torch.tensor(acc['t'])):
        raise RuntimeError('non-finite scoring loss')
    acc['ce'] = acc['t'] / acc['n']
    return acc if details else acc['ce']


@torch.no_grad()
def fit_v1(rows, prog, mode, rank):
    """Fit v1's program from block 0's attention input, with `prog` installed."""
    cap = {}

    def grab(mod, args, out):
        cap['x'] = args[0].reshape(-1, D).detach()
        cap['v'] = (out[1] if isinstance(out, tuple) else out).reshape(-1, D).detach()
        return None
    if mode == 'table':
        s = torch.zeros(50257, D, device=DEV)
        c = torch.zeros(50257, device=DEV)

        def collect(mod, args, out):
            grab(mod, args, out)
            t = STATE['idx'].reshape(-1)
            s.index_add_(0, t, cap['v'].float())
            c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
            return None
        sweep(rows, hooks=install(prog) + [H[0].attn.register_forward_hook(collect)])
        assert float(c.sum()) > 0, 'v1: no token counts'
        sn = c > 0
        tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
        tbl[sn] = s[sn] / c[sn].unsqueeze(1)
        return tbl
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect_l(mod, args, out):
        grab(mod, args, out)
        x = cap['x'].double()
        A.add_(x.T @ x); B.add_(x.T @ cap['v'].double()); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[0].attn.register_forward_hook(collect_l)])
    assert n['v'] > 0, 'v1: no fit positions'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    W = torch.linalg.solve(a + reg, B / n['v']).float()
    if rank < D:
        U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
        W = ((U[:, :rank] * S[:rank]) @ Vh[:rank]).float()
    return W


@torch.no_grad()
def main():
    global m, DEV, H, arm_gains, document_cluster_bootstrap, gain_structure_holds
    t0 = time.time()
    source = verify_source_binding()
    input_hashes, receipt = verify_inputs()
    model_identity = verify_model_binding()
    os.environ['HF_HUB_OFFLINE'] = '1'
    from whole_model_heldout_stats import (
        arm_gains as loaded_arm_gains,
        document_cluster_bootstrap as loaded_document_cluster_bootstrap,
        gain_structure_holds as loaded_gain_structure_holds,
    )
    from bilin18_joint_removal import m as loaded_model, DEV as loaded_device
    arm_gains = loaded_arm_gains
    document_cluster_bootstrap = loaded_document_cluster_bootstrap
    gain_structure_holds = loaded_gain_structure_holds
    m, DEV = loaded_model, loaded_device
    H = m.transformer.h
    K = torch.load(CONSTS, map_location='cpu')
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)
    print(f'WHOLE MODEL HELD-OUT | compile ONCE, score on {[n for n, _ in EVAL_SETS]} | '
          f'v1 tabled (§1698) | §1696 headline {S1696_BOTH:.2%}', flush=True)

    SPEC = (('simple', (1,), ()), ('attn_upgraded', ATTN_LAGS, ()),
            ('mlp_upgraded', (1,), S1672_MLP_TABLE_SITES),
            ('both', ATTN_LAGS, S1672_MLP_TABLE_SITES))
    progs, v1s = {}, {}
    for name, lags, tables in SPEC:
        CFG['lags'], CFG['tables'], CFG['v1'] = lags, tables, None
        V1P.pop('W', None)
        progs[name] = compile_stack(fit, ('mlp', 'attn'))
        v1s[name] = fit_v1(fit, progs[name], 'table', D)
        print(f'  compiled {name}', flush=True)
    del fit
    torch.cuda.empty_cache()

    out = {}
    exact_ceilings = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        CFG['lags'], CFG['tables'], CFG['v1'] = (1,), (), None
        V1P.pop('W', None)
        cl_record = ce(ev, seen, details=True)
        cl = cl_record['ce']
        if ename == 'skip7000':
            assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
                f'baseline CE {cl:.5f} disagrees with {S1683_CE_LIVE:.5f} (§1695)')
        identity_hooks = [H[L].mlp.register_forward_hook(identity_hook) for L in ALL18]
        identity_hooks += [H[L].attn.register_forward_hook(identity_hook) for L in ALL18]
        identity_ce = ce(ev, seen, hooks=identity_hooks)
        assert abs(identity_ce - cl) <= 1e-7, (
            f'36-site identity intervention changed CE: {identity_ce:.9f} vs {cl:.9f}')
        hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
              for L in ALL18]
        hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
               for L in ALL18]
        cc_record = ce(ev, seen, hooks=hs, details=True)
        if cc_record['counts'] != cl_record['counts']:
            raise RuntimeError(f'{ename}/constant: covered support drifted')
        cc = cc_record['ce']
        st = cc - cl
        assert torch.isfinite(torch.tensor(st)) and st > 0, f'invalid joint stake {st}'
        if ename == 'skip7000':
            assert abs(st - S1694_JOINT_STAKE) <= 1e-3, (
                f'joint stake {st:.5f} disagrees with {S1694_JOINT_STAKE:.5f}')
        row = {'ce_live': round(cl, 5), 'ce_constant': round(cc, 5),
               'stake': round(st, 5), 'identity_ce': round(identity_ce, 8),
               'covered_tokens': cl_record['n'],
               'possible_tokens': int(ev.shape[0] * (T - 64)),
               'coverage_rate': round(cl_record['n'] / (ev.shape[0] * (T - 64)), 7)}
        raw_records = {'live': cl_record, 'constant': cc_record}
        exact_values = {}
        print(f'  {ename:10s} CE live {cl:.5f} | joint stake {st:.4f} nats', flush=True)
        for name, lags, tables in SPEC:
            CFG['lags'], CFG['tables'] = lags, tables
            V1P['W'] = v1s[name]
            CFG['v1'] = 'table'
            arm_record = ce(ev, seen, hooks=install(progs[name]), details=True)
            if arm_record['counts'] != cl_record['counts']:
                raise RuntimeError(f'{ename}/{name}: covered support drifted across arms')
            raw_records[name] = arm_record
            ct = arm_record['ce']
            exact_values[name] = (cc - ct) / st
            row[name] = round(exact_values[name], 5)
            row[f'{name}_ce'] = round(ct, 5)
            print(f'      {name:14s} CEILING {row[name]:8.2%}', flush=True)
        CFG['v1'] = None
        V1P.pop('W', None)
        provenance_name = 'n192_' + ename
        document_ids = [e['document_id']
                        for e in receipt['document_provenance']['sets'][provenance_name]]
        row['cluster_bootstrap'] = document_cluster_bootstrap(
            raw_records, document_ids, draws=2000, seed=1699 if ename == 'skip7000' else 1700)
        row['raw_row_statistics'] = {
            name: {'loss_sums': record['loss_sums'],
                   'counts': record['counts']}
            for name, record in raw_records.items()
        }
        out[ename] = row
        exact_ceilings[ename] = exact_values
        del ev
        torch.cuda.empty_cache()

    ref, held = out['skip7000'], out['skip11000']
    ref_exact, held_exact = exact_ceilings['skip7000'], exact_ceilings['skip11000']
    names = [n for n, _, _ in SPEC]
    assert len(set(held[n] for n in names)) > 1, 'all held-out arms identical -- switch is a no-op'

    gr, gh = arm_gains(ref_exact), arm_gains(held_exact)

    pa = abs(held_exact['both'] - S1696_BOTH) <= 0.03
    pb = gain_structure_holds(gh, S1697_GAINS['both'])
    pc = (abs(ref_exact['both'] - S1696_BOTH) <= 0.005
          and abs(ref_exact['simple'] - 0.5094) <= 0.005
          and abs(ref['ce_live'] - S1683_CE_LIVE) <= 1e-3)
    held_boot = held['cluster_bootstrap']
    both_ci = held_boot['ceilings']['both']['ci95']
    joint_gain_ci = held_boot['gains']['both']['ci95']
    mlp_conditional_ci = held_boot['gains']['both_minus_attn']['ci95']
    attn_conditional_ci = held_boot['gains']['both_minus_mlp']['ci95']
    interaction_ci = held_boot['gains']['interaction']['ci95']
    pd = (both_ci[0] >= S1696_BOTH - 0.03 and both_ci[1] <= S1696_BOTH + 0.03
          and joint_gain_ci[0] > 0 and mlp_conditional_ci[0] > 0
          and attn_conditional_ci[0] > 0
          and interaction_ci[0] >= -0.01 and interaction_ci[1] <= 0.01)
    composition_replication_holds = pa and pb and pc and pd

    print(f'\n  ARM-BY-ARM reference -> held out:', flush=True)
    for n in names:
        print(f'    {n:14s} {ref[n]:7.2%} -> {held[n]:7.2%}   {held[n] - ref[n]:+.2%}', flush=True)
    print(f'  gains held out: attn {gh["attn"]:+.2%} | mlp {gh["mlp"]:+.2%} | both '
          f'{gh["both"]:+.2%} | singles sum {gh["attn"] + gh["mlp"]:+.2%}', flush=True)
    print(f'  (§1697 on skip7000: attn {S1697_GAINS["attn"]:+.2%} | mlp '
          f'{S1697_GAINS["mlp"]:+.2%} | both {S1697_GAINS["both"]:+.2%})', flush=True)
    print(f'  headline holds {pa} | gain structure holds {pb} | controls {pc} | '
          f'cluster precision holds {pd} | overall {composition_replication_holds}', flush=True)

    res = {'status': 'completed_prospective_conditional_composition_test',
           'authority': 'second_class_confirmation_only',
           'composition_replication_holds': bool(composition_replication_holds),
           'source': source,
           'model_identity': model_identity,
           'input_sha256': input_hashes,
           'config': {'sites': ALL18, 'ridge': RIDGE,
                      'eval_sets': [n for n, _ in EVAL_SETS],
                      'held_out': 'fineweb_n192_skip11000 -- the JOINT program has never been scored on it',
                      'arms': [{'name': n, 'lags': list(l), 'tables': list(t)} for n, l, t in SPEC],
                      'v1': 'per-token table on every arm (§1698)',
                      'compilation': 'INTERLEAVED bottom-up (§1669); compiled ONCE, scored on both evals',
                      'stake': 'recomputed per eval set (§1683 found it moving between these two)',
                      'pattern': 'prospective conditional composition replication; skip11000 was exposed to component-level work',
                      'coverage': 'mask pinned to n96_skip80', 'fit_rows': 'fineweb_n480_skip80.pt',
                      'uncertainty': 'paired 2000-draw FineWeb source-document cluster bootstrap',
                      's1696_both': S1696_BOTH, 's1697_gains': S1697_GAINS},
           'evals': out,
           'gains': {'reference': {k: round(v, 5) for k, v in gr.items()},
                     'held_out': {k: round(v, 5) for k, v in gh.items()}},
           'deltas': {n: round(held[n] - ref[n], 5) for n in names},
           'predictions': {'pred_a_headline_within_3pts': bool(pa),
                           'pred_b_gain_structure_holds': bool(pb),
                           'pred_c_controls_hold': bool(pc),
                           'pred_d_cluster_precision_holds': bool(pd),
                           'all_registered_predictions_hold': bool(composition_replication_holds)},
           'runtime_s': round(time.time() - t0, 1)}
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd} | '
          f'overall {composition_replication_holds}', flush=True)
    print(f'publishing create-only {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    write_create_only_atomic(OUT, res)
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if os.path.exists(OUT):
            raise
        failure = PT + f'whole_model_heldout_failure_{int(time.time())}.json'
        payload = {
            'status': 'failed_before_authoritative_result',
            'error_type': type(error).__name__,
            'error': str(error),
            'traceback': traceback.format_exc(),
            'source_sha256': sha256(__file__),
        }
        try:
            write_create_only_atomic(failure, payload)
        finally:
            raise
