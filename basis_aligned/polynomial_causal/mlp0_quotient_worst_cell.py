"""Authority-bound Stage-0 v1 screen for the existing MLP0 K=64 table.

This implements MLP0_CAUSAL_QUOTIENT_SPEC.md plus the prospective v1 amendment.
The abandoned skip-17000 development attempt produced no result.  V1 consumes only
the frozen, network-independent skip-21000 row receipt and writes a distinct output.
It retains document x 16-background-cell sufficient statistics for independent
rescoring; no pooled average can pass the equivalence gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path('/workspace/tensor_language')
BQ = ROOT / 'basis_aligned' / 'bilinear_quotient'
PC = ROOT / 'basis_aligned' / 'polynomial_causal'
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(PC))

from causal_response_quotient import (  # noqa: E402
    pointwise_dominates,
    score_worst_cell_equivalence,
)
from prepare_mlp0_quotient_stage0_v1_rows import (  # noqa: E402
    RECEIPT as ROW_RECEIPT,
    file_sha256,
    load_frozen_role,
    load_frozen_rows,
)


D = 1152
T = 256
V = 50257
NFIT = 960
NEVAL = 192
FIT_SKIP = 80
EVAL_SKIP = 21000
BATCH = 8
K = 64
OUT = BQ / 'mlp0_quotient_stage0_v1_results.json'
AUTHORITY = BQ / 'mlp0_quotient_stage0_v1_collector_authority.json'
FIT_RECEIPT = BQ / 'mlp0_quotient_stage0_v1_fit_receipt.json'
FAILURE = BQ / 'mlp0_quotient_stage0_v1_failure.json'
LOCK = Path('/workspace/runs/.bilin18_mlp0_quotient_stage0_v1.lock')
MARGINS = {'kl': 0.01, 'ce': 0.0075, 'attn1_nrmse': 0.05, 'mlp1_nrmse': 0.05}
CELL_NAMES = [
    f"pos{pos}_freq{freq}_prev{prev}_dev{dev}"
    for pos in range(2) for freq in range(2) for prev in range(2) for dev in range(2)
]
DEV = 'cuda'
m = None
H = None
torch.manual_seed(0)

STATE = {'arm': 'O', 'idx': None, 'tables': {}, 'caps': {}, 'block0_stream': None}


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(payload, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(payload, indent=1) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_authority() -> dict:
    if not AUTHORITY.is_file():
        raise RuntimeError(f'collector authority absent: {AUTHORITY}')
    authority = json.loads(AUTHORITY.read_text())
    if (authority.get('status') != 'frozen_before_any_v1_model_forward'
            or authority.get('output_path') != str(OUT)
            or authority.get('row_receipt_sha256') != file_sha256(ROW_RECEIPT)
            or authority.get('fit_receipt_sha256') != file_sha256(FIT_RECEIPT)):
        raise RuntimeError('collector authority identity/status mismatch')
    for raw, expected in authority.get('source_hashes', {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f'authority-bound source changed: {raw}')
    if OUT.exists() or FAILURE.exists():
        raise RuntimeError('v1 output namespace is already spent')
    return authority


def acquire_lock() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    return os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def load_model() -> None:
    global m, H
    from bilin18_joint_removal import m as loaded
    m = loaded
    H = m.transformer.h


def block0_pre_hook(module, args):
    x, _, x0 = args
    STATE['block0_stream'] = (
        module.lambdas[0] * x + module.lambdas[1] * x0
    ).detach().float()


def attn0_hook(module, args, output):
    value = output[0] if isinstance(output, tuple) else output
    stream = STATE.get('block0_stream')
    if stream is None:
        raise RuntimeError('block-0 stream capture missing')
    STATE['caps']['pre_mlp0'] = stream + value.detach().float()


def m0_hook(module, args, output):
    STATE['caps']['m0'] = output.detach().float()
    arm = STATE['arm']
    if arm == 'O':
        return None
    idx = STATE['idx']
    if arm == 'M':
        replacement = STATE['tables']['M'].expand(idx.shape[0], idx.shape[1], D)
    else:
        replacement = STATE['tables'][arm][idx]
    return replacement.to(output.dtype)


def attn1_hook(module, args, output):
    value = output[0] if isinstance(output, tuple) else output
    STATE['caps']['attn1'] = value.detach().float()
    return None


def mlp1_hook(module, args, output):
    STATE['caps']['mlp1'] = output.detach().float()
    return None


@torch.no_grad()
def fwd(idx: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if m is None or H is None:
        raise RuntimeError('model was not authority-gated and loaded')
    STATE['arm'] = arm
    STATE['idx'] = idx
    STATE['caps'] = {}
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x
    v1 = None
    for block in H:
        x, v1 = block(x, v1, x0)
    logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    required = {'pre_mlp0', 'm0', 'attn1', 'mlp1'}
    if set(STATE['caps']) != required:
        raise RuntimeError(f"capture failure: got {sorted(STATE['caps'])}")
    return logits.float(), {name: value.clone() for name, value in STATE['caps'].items()}


def build_token_tables(fit_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    token_sum = torch.zeros(V, D, device=DEV)
    token_count = torch.zeros(V, device=DEV)
    attn_sum = attn_sq = mlp_sum = mlp_sq = 0.0
    n_vector = 0
    for start in range(0, NFIT, BATCH):
        idx = fit_rows[start:start + BATCH, :-1].to(DEV).contiguous()
        _, cap = fwd(idx, 'O')
        flat_token = idx.reshape(-1)
        flat_m0 = cap['m0'].reshape(-1, D)
        token_sum.index_add_(0, flat_token, flat_m0)
        token_count.index_add_(0, flat_token, torch.ones_like(flat_token, dtype=torch.float))
        for name, prefix in (('attn1', 'attn'), ('mlp1', 'mlp')):
            value = cap[name].reshape(-1, D).double()
            if prefix == 'attn':
                attn_sum += float(value.sum())
                attn_sq += float((value * value).sum())
            else:
                mlp_sum += float(value.sum())
                mlp_sq += float((value * value).sum())
        n_vector += flat_m0.shape[0] * D

    global_mean = token_sum.sum(0) / token_count.sum()
    token_table = torch.where(
        token_count.unsqueeze(1) > 0,
        token_sum / token_count.clamp_min(1).unsqueeze(1),
        global_mean.unsqueeze(0),
    )
    scales = {
        'attn1': math.sqrt(max(attn_sq / n_vector - (attn_sum / n_vector) ** 2, 1e-20)),
        'mlp1': math.sqrt(max(mlp_sq / n_vector - (mlp_sum / n_vector) ** 2, 1e-20)),
    }
    return token_table, token_count, scales


@torch.no_grad()
def build_cluster_table(token_table: torch.Tensor, token_count: torch.Tensor,
                        global_mean: torch.Tensor, *, downstream: bool
                        ) -> tuple[torch.Tensor, torch.Tensor, int]:
    centered = token_table - global_mean
    if downstream:
        maps = [H[1].attn.c_q.weight, H[1].attn.c_k.weight, H[1].attn.c_v.weight,
                H[1].mlp.Left.weight, H[1].mlp.Right.weight]
        generator = torch.Generator().manual_seed(13)
        parts = []
        for weight in maps:
            image = centered @ weight.float().to(DEV).T
            projection = torch.randn(
                image.shape[1], 128, generator=generator, device='cpu'
            ).to(DEV) / math.sqrt(image.shape[1])
            parts.append(image @ projection)
        embedding = F.normalize(torch.cat(parts, 1), dim=1)
        seed = 100 + K
    else:
        embedding = F.normalize(centered, dim=1)
        seed = 200 + K

    generator = torch.Generator().manual_seed(seed)
    weights = token_count.clamp_min(0)
    centroids = embedding[torch.randperm(embedding.shape[0], generator=generator)[:K]].clone()
    for _ in range(20):
        labels = torch.cdist(embedding, centroids).argmin(1)
        for cluster in range(K):
            selected = labels == cluster
            mass = weights[selected].sum()
            if float(mass) > 0:
                centroids[cluster] = (
                    embedding[selected] * weights[selected].unsqueeze(1)
                ).sum(0) / mass
    labels = torch.cdist(embedding, centroids).argmin(1)
    table_sum = torch.zeros(K, D, device=DEV)
    table_weight = torch.zeros(K, device=DEV)
    table_sum.index_add_(0, labels, token_table * weights.unsqueeze(1))
    table_weight.index_add_(0, labels, weights)
    compact = torch.where(
        table_weight.unsqueeze(1) > 0,
        table_sum / table_weight.clamp_min(1e-6).unsqueeze(1),
        global_mean.unsqueeze(0),
    )
    occupied = int((table_weight > 0).sum())
    return compact[labels], labels, occupied


def punctuation_table() -> torch.Tensor:
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    values = torch.zeros(V, dtype=torch.bool)
    for token in range(V):
        raw = encoder.decode([token])
        stripped = raw.strip()
        values[token] = bool(
            '\n' in raw or stripped == '' or re.fullmatch(r"[^\w\s]+", stripped)
        )
    return values


def empty_measurements() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    sums = {name: torch.zeros(NEVAL, 16, dtype=torch.float64) for name in MARGINS}
    counts = {name: torch.zeros(NEVAL, 16, dtype=torch.float64) for name in MARGINS}
    return sums, counts


def add_cells(sums, counts, document_offset, cell, mask, effects):
    for row in range(mask.shape[0]):
        for cell_id in range(16):
            selected = mask[row] & (cell[row] == cell_id)
            count = int(selected.sum())
            if count:
                counts[document_offset + row, cell_id] += count
                sums[document_offset + row, cell_id] += float(effects[row][selected].sum())


def compute_effects(outputs, target, direct_scales, reference, candidate, *,
                    ce_reference=None, ce_candidate=None):
    """Compute one contrast with independently registered KL and CE orientation."""
    ref_logits, ref_cap = outputs[reference]
    can_logits, can_cap = outputs[candidate]
    ref_logp = F.log_softmax(ref_logits, dim=-1)
    can_logp = F.log_softmax(can_logits, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - can_logp)).sum(-1)
    ce_reference = reference if ce_reference is None else ce_reference
    ce_candidate = candidate if ce_candidate is None else ce_candidate
    ce_ref_logits = outputs[ce_reference][0]
    ce_can_logits = outputs[ce_candidate][0]
    ce_ref = F.cross_entropy(
        ce_ref_logits.reshape(-1, ce_ref_logits.shape[-1]),
        target.reshape(-1), reduction='none'
    ).view_as(target)
    ce_can = F.cross_entropy(
        ce_can_logits.reshape(-1, ce_can_logits.shape[-1]),
        target.reshape(-1), reduction='none'
    ).view_as(target)
    attn = ((can_cap['attn1'] - ref_cap['attn1']).pow(2).mean(-1).sqrt()
            / direct_scales['attn1'])
    mlp = ((can_cap['mlp1'] - ref_cap['mlp1']).pow(2).mean(-1).sqrt()
           / direct_scales['mlp1'])
    return {'kl': kl, 'ce': ce_can - ce_ref,
            'attn1_nrmse': attn, 'mlp1_nrmse': mlp}


def fit_state(fit_rows: torch.Tensor) -> dict:
    token_table, token_count, direct_scales = build_token_tables(fit_rows)
    global_mean = token_table[token_count > 0]
    global_mean = (
        global_mean * token_count[token_count > 0].unsqueeze(1)
    ).sum(0) / token_count.sum()
    q64, q64_labels, q64_occupied = build_cluster_table(
        token_table, token_count, global_mean, downstream=True
    )
    a64, a64_labels, a64_occupied = build_cluster_table(
        token_table, token_count, global_mean, downstream=False
    )
    fit_tokens = fit_rows[:, :-1].reshape(-1).to(DEV)
    frequency_median = float(token_count[fit_tokens].median())
    pre_mlp0_norms = []
    for start in range(0, NFIT, BATCH):
        idx = fit_rows[start:start + BATCH, :-1].to(DEV).contiguous()
        _, cap = fwd(idx, 'O')
        pre_mlp0_norms.append(cap['pre_mlp0'].norm(dim=-1).reshape(-1).cpu())
    pre_mlp0_norm_median = float(torch.cat(pre_mlp0_norms).median())
    return {
        'token_table': token_table, 'token_count': token_count,
        'global_mean': global_mean, 'q64': q64, 'a64': a64,
        'q64_labels': q64_labels, 'a64_labels': a64_labels,
        'q64_occupied': q64_occupied, 'a64_occupied': a64_occupied,
        'direct_scales': direct_scales, 'frequency_median': frequency_median,
        'pre_mlp0_norm_median': pre_mlp0_norm_median,
    }


def fit_receipt_payload(state: dict, fit_rows: torch.Tensor) -> dict:
    return {
        'schema_version': 1,
        'receipt_kind': 'mlp0_quotient_stage0_v1_fit_constants',
        'status': 'frozen_before_any_v1_evaluation_model_forward',
        'fit_rows_sha256': tensor_hash(fit_rows),
        'fit_rows_receipt_sha256': file_sha256(ROW_RECEIPT),
        'constants': {
            'frequency_median': state['frequency_median'],
            'pre_mlp0_raw_residual_norm_median': state['pre_mlp0_norm_median'],
            'direct_scales': state['direct_scales'],
            'token_table_sha256': tensor_hash(state['token_table']),
            'q64_table_sha256': tensor_hash(state['q64']),
            'a64_table_sha256': tensor_hash(state['a64']),
            'q64_assignments_sha256': tensor_hash(state['q64_labels']),
            'a64_assignments_sha256': tensor_hash(state['a64_labels']),
            'q64_occupied_positive_mass_clusters': state['q64_occupied'],
            'a64_occupied_positive_mass_clusters': state['a64_occupied'],
        },
    }


def register_hooks():
    assert H is not None
    return [
        H[0].register_forward_pre_hook(block0_pre_hook),
        H[0].attn.register_forward_hook(attn0_hook),
        H[0].mlp.register_forward_hook(m0_hook),
        H[1].attn.register_forward_hook(attn1_hook),
        H[1].mlp.register_forward_hook(mlp1_hook),
    ]


@torch.no_grad()
def freeze_fit_receipt() -> None:
    if FIT_RECEIPT.exists() or AUTHORITY.exists() or OUT.exists() or FAILURE.exists():
        raise RuntimeError('v1 fit/collector namespace is already spent')
    _, fit_full = load_frozen_role('fit')
    fit_rows = fit_full[:, :T + 1].contiguous()
    load_model()
    hooks = register_hooks()
    try:
        state = fit_state(fit_rows)
        payload = fit_receipt_payload(state, fit_rows)
        write_json_atomic(payload, FIT_RECEIPT)
        print(json.dumps(payload, indent=1), flush=True)
        print(f'wrote {FIT_RECEIPT}', flush=True)
    finally:
        STATE['arm'] = 'O'
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def main() -> None:
    started = time.time()
    authority = validate_authority()
    lock_fd = acquire_lock()
    row_receipt, frozen = load_frozen_rows()
    fit_rows = frozen['fit'][:, :T + 1].contiguous()
    eval_rows = frozen['eval'][:, :T + 1].contiguous()
    fit_row_set = {tuple(value.tolist()) for value in fit_rows}
    if any(tuple(row.tolist()) in fit_row_set for row in eval_rows):
        raise RuntimeError('fit/evaluation row overlap')
    load_model()
    assert H is not None

    hooks = register_hooks()
    try:
        state = fit_state(fit_rows)
        expected_fit = json.loads(FIT_RECEIPT.read_text())
        observed_fit = fit_receipt_payload(state, fit_rows)
        if observed_fit != expected_fit:
            raise RuntimeError('fit-frozen constants/tables changed before evaluation')
        token_table, token_count = state['token_table'], state['token_count']
        global_mean, direct_scales = state['global_mean'], state['direct_scales']
        q64, a64 = state['q64'], state['a64']
        q64_labels, a64_labels = state['q64_labels'], state['a64_labels']
        q64_occupied, a64_occupied = state['q64_occupied'], state['a64_occupied']
        frequency_median = state['frequency_median']
        pre_mlp0_norm_median = state['pre_mlp0_norm_median']
        STATE['tables']['T'] = token_table
        STATE['tables']['Q64'] = q64
        STATE['tables']['A64'] = a64
        STATE['tables']['M'] = global_mean.view(1, 1, D)

        punctuation = punctuation_table().to(DEV)
        contrasts = {
            'T_vs_O': empty_measurements(),
            'Q64_vs_T': empty_measurements(),
            'A64_vs_T': empty_measurements(),
            'M_vs_T': empty_measurements(),
        }
        covered_positions = total_positions = 0

        for start in range(0, NEVAL, BATCH):
            batch = eval_rows[start:start + BATCH].to(DEV)
            idx, target = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            outputs = {}
            for arm in ('O', 'T', 'Q64', 'A64', 'M'):
                outputs[arm] = fwd(idx, arm)

            position = torch.arange(T, device=DEV).view(1, T).expand_as(idx)
            previous = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], dim=1)
            prev_boundary = previous < 0
            previous_safe = previous.clamp_min(0)
            prev_kind = prev_boundary | punctuation[previous_safe]
            frequency_kind = token_count[idx] > frequency_median
            pre_mlp0_norm = outputs['O'][1]['pre_mlp0'].norm(dim=-1)
            deviation_kind = pre_mlp0_norm > pre_mlp0_norm_median
            position_kind = position >= (T // 2)
            cell = (position_kind.long() * 8 + frequency_kind.long() * 4
                    + prev_kind.long() * 2 + deviation_kind.long())
            covered = token_count[idx] > 0
            valid = covered
            covered_positions += int(valid.sum())
            total_positions += int(torch.ones_like(valid).sum())

            definitions = {
                # Registered fine-interface reference: KL(T||O), CE_T-CE_O.
                'T_vs_O': ('T', 'O', 'O', 'T'),
                'Q64_vs_T': ('T', 'Q64', 'T', 'Q64'),
                'A64_vs_T': ('T', 'A64', 'T', 'A64'),
                'M_vs_T': ('T', 'M', 'T', 'M'),
            }
            for name, (reference, candidate, ce_reference, ce_candidate) in definitions.items():
                sums, counts = contrasts[name]
                values = compute_effects(
                    outputs, target, direct_scales, reference, candidate,
                    ce_reference=ce_reference, ce_candidate=ce_candidate
                )
                for consumer, value in values.items():
                    add_cells(sums[consumer], counts[consumer], start, cell, valid, value)
            print(f'eval {min(start + BATCH, NEVAL)}/{NEVAL}', flush=True)

        coverage = covered_positions / max(total_positions, 1)
        reports = {}
        for name, (sums, counts) in contrasts.items():
            reports[name] = score_worst_cell_equivalence(
                {key: value.numpy() for key, value in sums.items()},
                {key: value.numpy() for key, value in counts.items()},
                margins=MARGINS,
                cell_names=CELL_NAMES,
                minimum_documents_per_cell=30,
                n_bootstrap=10_000,
                seed=20260827,
            )

        sensitivity = {}
        for consumer, report in reports['M_vs_T']['consumers'].items():
            sensitivity[consumer] = max(report['cell_standardized_effects'].values()) > 1
        gates = {
            'coverage_ge_90pct': coverage >= 0.90,
            'token_table_vs_live': reports['T_vs_O']['equivalence_passes'],
            'q64_vs_token_table': reports['Q64_vs_T']['equivalence_passes'],
            'q64_pointwise_beats_a64': pointwise_dominates(
                reports['Q64_vs_T'], reports['A64_vs_T']
            ),
            'mean_assay_sensitive_all_consumers': all(sensitivity.values()),
        }
        gates['stage0_passes'] = all(gates.values())
        result = {
            'schema_version': 1,
            'experiment': 'mlp0_quotient_worst_cell_stage0',
            'rows': {
                'fit': {'n': NFIT, 'skip': FIT_SKIP, 'sha256': tensor_hash(fit_rows)},
                'eval': {'n': NEVAL, 'skip': EVAL_SKIP, 'sha256': tensor_hash(eval_rows)},
                'exact_row_overlap': False,
                'receipt_path': str(ROW_RECEIPT),
                'receipt_sha256': file_sha256(ROW_RECEIPT),
            },
            'construction': {'K': K, 'downstream_projection_seed': 13,
                             'downstream_kmeans_seed': 100 + K,
                             'activation_kmeans_seed': 200 + K,
                             'q64_occupied_positive_mass_clusters': q64_occupied,
                             'a64_occupied_positive_mass_clusters': a64_occupied,
                             'token_table_sha256': tensor_hash(token_table),
                             'q64_table_sha256': tensor_hash(q64),
                             'a64_table_sha256': tensor_hash(a64),
                             'q64_assignments_sha256': tensor_hash(q64_labels),
                             'a64_assignments_sha256': tensor_hash(a64_labels)},
            'fit_frozen': {'frequency_median': frequency_median,
                           'pre_mlp0_raw_residual_norm_median': pre_mlp0_norm_median,
                           'direct_scales': direct_scales},
            'coverage': coverage,
            'margins': MARGINS,
            'cell_names': CELL_NAMES,
            'reports': reports,
            'mean_sensitivity_by_consumer': sensitivity,
            'gates': gates,
            'authority': authority,
            'sufficient_statistics': {
                contrast: {
                    consumer: {'sums': sums[consumer].tolist(),
                               'counts': counts[consumer].tolist()}
                    for consumer in MARGINS
                }
                for contrast, (sums, counts) in contrasts.items()
            },
            'interpretation': (
                'A pass licenses only a finite 16-cell global-deployment screen; '
                'it does not establish arbitrary-background causal equivalence.'
            ),
            'runtime_s': time.time() - started,
        }
        write_json_atomic(result, OUT)
        print(json.dumps({'coverage': coverage, 'gates': gates}, indent=1), flush=True)
        print(f'wrote {OUT} ({result["runtime_s"]:.1f}s)', flush=True)
    finally:
        STATE['arm'] = 'O'
        for hook in hooks:
            hook.remove()
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    if '--freeze-fit-receipt' in sys.argv:
        freeze_fit_receipt()
    else:
        main()
