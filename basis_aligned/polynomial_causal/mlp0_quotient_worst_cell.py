"""Fresh-row worst-cell screen for the existing MLP0 reader-defined K=64 table.

This is the Stage-0 experiment preregistered in MLP0_CAUSAL_QUOTIENT_SPEC.md.  It
reuses the exact token-table/downstream-reader construction from
bilinear_quotient/mlp0_downstream_clusters.py, globally deploys five arms, and keeps
document x 16-background-cell sums for four consumers.  No pooled average can pass
the equivalence gate.
"""

from __future__ import annotations

import hashlib
import json
import math
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

from bilin18_joint_removal import DEV, m  # noqa: E402
import census_lib as cl  # noqa: E402
from causal_response_quotient import (  # noqa: E402
    pointwise_dominates,
    score_worst_cell_equivalence,
)


D = 1152
T = 256
V = 50257
NFIT = 960
NEVAL = 192
FIT_SKIP = 80
EVAL_SKIP = 17000
BATCH = 8
K = 64
OUT = BQ / 'mlp0_quotient_worst_cell_results.json'
MARGINS = {'kl': 0.01, 'ce': 0.0075, 'attn1_nrmse': 0.05, 'mlp1_nrmse': 0.05}
CELL_NAMES = [
    f"pos{pos}_freq{freq}_prev{prev}_dev{dev}"
    for pos in range(2) for freq in range(2) for prev in range(2) for dev in range(2)
]
H = m.transformer.h
torch.manual_seed(0)

STATE = {'arm': 'O', 'idx': None, 'tables': {}, 'caps': {}}


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


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
    STATE['arm'] = arm
    STATE['idx'] = idx
    STATE['caps'] = {}
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x
    v1 = None
    for block in H:
        x, v1 = block(x, v1, x0)
    logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    required = {'m0', 'attn1', 'mlp1'}
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
                        global_mean: torch.Tensor, *, downstream: bool) -> torch.Tensor:
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
    return compact[labels]


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


@torch.no_grad()
def main() -> None:
    started = time.time()
    cl.use_state(str(BQ / 'census_state_diverse.pt'))
    fit_rows = cl.fineweb_rows(NFIT, skip=FIT_SKIP)[:, :T + 1].contiguous()
    eval_rows = cl.fineweb_rows(NEVAL, skip=EVAL_SKIP)[:, :T + 1].contiguous()
    fit_row_set = {tuple(value.tolist()) for value in fit_rows}
    if any(tuple(row.tolist()) in fit_row_set for row in eval_rows):
        raise RuntimeError('fit/evaluation row overlap')

    hooks = [
        H[0].mlp.register_forward_hook(m0_hook),
        H[1].attn.register_forward_hook(attn1_hook),
        H[1].mlp.register_forward_hook(mlp1_hook),
    ]
    try:
        token_table, token_count, direct_scales = build_token_tables(fit_rows)
        global_mean = token_table[token_count > 0]
        global_mean = (
            global_mean * token_count[token_count > 0].unsqueeze(1)
        ).sum(0) / token_count.sum()
        STATE['tables']['T'] = token_table
        STATE['tables']['Q64'] = build_cluster_table(
            token_table, token_count, global_mean, downstream=True
        )
        STATE['tables']['A64'] = build_cluster_table(
            token_table, token_count, global_mean, downstream=False
        )
        STATE['tables']['M'] = global_mean.view(1, 1, D)

        fit_tokens = fit_rows[:, :-1].reshape(-1).to(DEV)
        frequency_median = float(token_count[fit_tokens].median())
        deviation_norms = []
        for start in range(0, NFIT, BATCH):
            idx = fit_rows[start:start + BATCH, :-1].to(DEV).contiguous()
            _, cap = fwd(idx, 'O')
            deviation_norms.append(
                (cap['m0'] - token_table[idx]).float().norm(dim=-1).reshape(-1).cpu()
            )
        deviation_median = float(torch.cat(deviation_norms).median())

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
            deviation = (outputs['O'][1]['m0'] - token_table[idx]).norm(dim=-1)
            deviation_kind = deviation > deviation_median
            position_kind = position >= (T // 2)
            cell = (position_kind.long() * 8 + frequency_kind.long() * 4
                    + prev_kind.long() * 2 + deviation_kind.long())
            covered = token_count[idx] > 0
            valid = covered & (position >= 64)
            covered_positions += int(valid.sum())
            total_positions += int((position >= 64).sum())

            def effects(reference, candidate):
                ref_logits, ref_cap = outputs[reference]
                can_logits, can_cap = outputs[candidate]
                ref_logp = F.log_softmax(ref_logits, dim=-1)
                can_logp = F.log_softmax(can_logits, dim=-1)
                ref_p = ref_logp.exp()
                kl = (ref_p * (ref_logp - can_logp)).sum(-1)
                ce_ref = F.cross_entropy(
                    ref_logits.reshape(-1, V), target.reshape(-1), reduction='none'
                ).view_as(target)
                ce_can = F.cross_entropy(
                    can_logits.reshape(-1, V), target.reshape(-1), reduction='none'
                ).view_as(target)
                attn = ((can_cap['attn1'] - ref_cap['attn1']).pow(2).mean(-1).sqrt()
                        / direct_scales['attn1'])
                mlp = ((can_cap['mlp1'] - ref_cap['mlp1']).pow(2).mean(-1).sqrt()
                       / direct_scales['mlp1'])
                return {'kl': kl, 'ce': ce_can - ce_ref,
                        'attn1_nrmse': attn, 'mlp1_nrmse': mlp}

            definitions = {
                'T_vs_O': ('O', 'T'),
                'Q64_vs_T': ('T', 'Q64'),
                'A64_vs_T': ('T', 'A64'),
                'M_vs_T': ('T', 'M'),
            }
            for name, (reference, candidate) in definitions.items():
                sums, counts = contrasts[name]
                for consumer, value in effects(reference, candidate).items():
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
            },
            'construction': {'K': K, 'downstream_projection_seed': 13,
                             'downstream_kmeans_seed': 100 + K,
                             'activation_kmeans_seed': 200 + K},
            'fit_frozen': {'frequency_median': frequency_median,
                           'context_deviation_norm_median': deviation_median,
                           'direct_scales': direct_scales},
            'coverage': coverage,
            'margins': MARGINS,
            'cell_names': CELL_NAMES,
            'reports': reports,
            'mean_sensitivity_by_consumer': sensitivity,
            'gates': gates,
            'interpretation': (
                'A pass licenses only a finite 16-cell global-deployment screen; '
                'it does not establish arbitrary-background causal equivalence.'
            ),
            'runtime_s': time.time() - started,
        }
        OUT.write_text(json.dumps(result, indent=1) + '\n')
        print(json.dumps({'coverage': coverage, 'gates': gates}, indent=1), flush=True)
        print(f'wrote {OUT} ({result["runtime_s"]:.1f}s)', flush=True)
    finally:
        STATE['arm'] = 'O'
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    main()
