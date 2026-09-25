"""Bounded CPU review control; reads frozen receipts, never invokes the model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'basis_aligned/polynomial_causal'
A = ROOT / 'basis_aligned/bilinear_quotient/circuits/followups'
OUT = P / 'REVIEW_READER_IDENTIFIABILITY_2026-09-20_1337.json'

def main():
    assert not OUT.exists()
    torch.set_num_threads(4)
    start = time.perf_counter()
    paths = [A / 'residual_reader_transfer_v1_tensors.pt',
             A / 'residual_reader_transfer_v1_result.json',
             A / 'source_ood_v3_role_bank_v1_result.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    bundle = torch.load(paths[0], map_location='cpu', weights_only=False)
    prior = json.loads(paths[1].read_text())
    fresh = json.loads(paths[2].read_text())
    spans = []
    for role in ['subject', 'attractor']:
        train = [g for g in bundle['groups'] if g['panel'] == 'opposite' and g['role'] == role]
        held = [g for g in bundle['groups'] if g['panel'] == 'congruent' and g['role'] == role]
        X = torch.cat([g['sources'].reshape(-1, 1152) for g in train]).double()
        Z = torch.cat([g['sources'].reshape(-1, 1152) for g in held]).double()
        U, s, Vh = torch.linalg.svd(X, full_matrices=False)
        rank = int((s > s[0] * 1e-10).sum())
        V = Vh[:rank].T
        residual = Z - (Z @ V) @ V.T
        ix = int(residual.norm(dim=1).argmax())
        w = residual[ix] / residual[ix].norm()
        train_null = float((X @ w).norm() / X.norm())
        held_signal = float((Z @ w).norm() / Z.norm())
        Y = torch.cat([torch.einsum('bkd,bod->bko', g['sources'], g['reader']).reshape(-1, 9) for g in train])
        # Independent full-capacity least-squares rescue: no held responses used.
        B = V @ ((U[:, :rank].T @ Y) / s[:rank, None])
        train_fit = float((X @ B - Y).norm() / Y.norm())
        heldY = torch.cat([torch.einsum('bkd,bod->bko', g['sources'], g['reader']).reshape(-1, 9) for g in held])
        held_fit = float((Z @ B - heldY).norm() / heldY.norm())
        planted = torch.arange(1152 * 2, dtype=torch.float64).reshape(1152, 2).sin()
        plantedY = X @ planted
        recovered = V @ ((U[:, :rank].T @ plantedY) / s[:rank, None])
        planted_error = float((X @ recovered - plantedY).norm() / plantedY.norm())
        spans.append(dict(role=role, shape=list(X.shape), rank_rtol_1e10=rank,
            nullity=1152-rank, held_outside_span_fraction=float(residual.norm()/Z.norm()),
            train_null_relative=train_null, held_unit_null_signal_relative=held_signal,
            train_ls_relative=train_fit, held_ls_relative=held_fit,
            planted_training_recovery=planted_error, condition_retained=float(s[0]/s[rank-1])))
        assert train_null < 1e-9 and planted_error < 1e-8
    partitions = []
    for record in prior['records']:
        if record['arm'] != 'role_bank':
            continue
        role, family = record['role'], record['family']
        group = next(g for g in bundle['groups'] if g['panel']=='congruent' and g['role']==role and family in g['families'])
        ids = [i for i, f in enumerate(group['families']) if f == family]
        D = group['sources'][ids]
        R = group['reader'][ids].clone()
        RB = bundle['role_banks'][role][None].expand(len(ids), -1, -1).clone()
        R[:, 0] *= group['orientation'][ids, None]
        RB[:, 0] *= group['orientation'][ids, None]
        aa = torch.tensor(record['amplitudes'], dtype=torch.float64)
        tangent = -torch.einsum('bod,bkd,bk->bo', R, D, aa)
        bank = -torch.einsum('bod,bkd,bk->bo', RB, D, aa)
        effect = torch.tensor(record['effect'], dtype=torch.float64)
        transfer, curvature, total = bank-tangent, tangent-effect, bank-effect
        closure = float((transfer+curvature-total).abs().max())
        den = effect[:, 0].norm()
        partitions.append(dict(role=role, family=family, number_effect_norm=float(den),
            bank_error=float(total[:,0].norm()/den),
            exact_tangent_error=float(curvature[:,0].norm()/den),
            reader_transfer_error=float(transfer[:,0].norm()/den),
            signed_transfer_share=float((transfer[:,0]@total[:,0])/total[:,0].square().sum()),
            closure=closure))
        assert closure < 1e-12
    summaries=[]
    for role in ['subject', 'attractor']:
        rows=[r for r in fresh['records'] if r['dataset']!='v2_replay' and r['role']==role]
        summaries.append(dict(role=role, cells=len(rows),
            selective_pass=sum(r['retention']>=.8 and r['control_ratio']<=.1 for r in rows),
            predictive_pass=sum(r['prediction_errors'][0]<=.1 and max(r['prediction_errors'][1:])<=.05 for r in rows),
            capability_pass=sum(r['capability']>=.9 for r in rows),
            worst_number_prediction=max(r['prediction_errors'][0] for r in rows)))
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        scope='Opened v2 CPU algebra/identifiability; v3 receipt scoring only. No new native execution or fit adoption.',
        spans=spans, finite_error_partition=partitions, fresh_v3=summaries,
        seconds=time.perf_counter()-start)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
