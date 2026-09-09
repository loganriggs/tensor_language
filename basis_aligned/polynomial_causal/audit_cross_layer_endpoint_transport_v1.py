"""Bounded CPU screen of a complete conditional endpoint write/read path."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language'); POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY)]
import torch
import exact_source_edit_reference as E
import forward_endpoint_program_reference as F
import frozen_query_read_reference as Q
import query_hop_fork_reference as H
import suffix_join_middle_match_reference as R
import field_intervention_metrics as M


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def center(x): return x-x.mean(-1, keepdim=True)
def rms(x): return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2); signal.alarm(180); started = time.perf_counter()
    out = POLY/'CROSS_LAYER_ENDPOINT_TRANSPORT_V1_RESULT.json'
    rows = POLY/'CROSS_LAYER_ENDPOINT_TRANSPORT_V1_ROWS.pt'
    assert not out.exists() and not rows.exists(); controls = H.controls(); assert controls['passed']
    package = POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package) == 'e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program = E.load_package(torch.load(package, map_location='cpu', weights_only=True))
    audits = []; saved = {}; results = {}; live = True
    with torch.inference_mode():
        for pop, (tokens, masks, metadata) in R.populations().items():
            values = {k: [] for k in ('write', 'path', 'removal', 'native_query_logits')}; meta = []
            for i in range(0, len(tokens), 4):
                if metadata[i]['orientation'] != 'B2_later': continue
                tok, current = H.fork(tokens[[i, i+3]], [metadata[i], metadata[i+3]])
                mask = masks[i:i+1].expand(len(tok), -1, -1).clone(); mask[:, :, :48:2] = False
                assert bool((mask.sum((1, 2)) == 2).all())
                context = program.prepare(tok); write = F.messages(program.background, tok, mask)
                audits.append(M.correspondence(write, F.native_messages(program.background, tok, mask)))
                path = Q.read(program, context, write)
                full = context['logits'][:, -1]-.5*program.background.head(context['x'][:, -1])
                audits.append(M.correspondence(Q.read(program, context, context['x']), full))
                audits.append(M.correspondence(Q.read(program, context, write*.5), path*.5))
                removal = context['logits'][:, -1]-program.edit(context, -write)[:, -1]
                live &= rms(path) > 1e-6
                for k, v in (('write', write), ('path', path), ('removal', removal),
                             ('native_query_logits', context['logits'][:, -1])):
                    values[k].append(v.clone())
                meta.extend(current)
            values = {k: torch.cat(v) for k, v in values.items()}; groups = {}
            for hop in range(4):
                base = torch.tensor([m['case'] == 'base' and m['hop'] == hop for m in meta])
                donor = torch.tensor([m['case'] == 'both' and m['hop'] == hop for m in meta])
                path, other = center(values['path'][base]), center(values['path'][donor])
                select = base | donor; removal = center(values['removal'][select])
                change = rms(other-path)/max(rms(path), 1e-6)
                causal = rms(center(values['path'][select])-removal)/max(rms(removal), 1e-6)
                cosine = (path*other).sum(-1)/(path.norm(dim=-1)*other.norm(dim=-1)).clamp_min(1e-12)
                wp, wd = values['write'][base], values['write'][donor]
                wc = (wp*wd).sum((1, 2))/(wp.norm(dim=(1, 2))*wd.norm(dim=(1, 2))).clamp_min(1e-12)
                groups[str(hop)] = {'pairs': int(base.sum()), 'base_path_rms': rms(path),
                    'path_relative_change': change, 'writer_relative_change': rms(wd-wp)/max(rms(wp), 1e-6),
                    'path_cosine_mean': float(cosine.mean()), 'path_negative_cosine_count': int((cosine < 0).sum()),
                    'writer_negative_cosine_count': int((wc < 0).sum()), 'removal_rms': rms(removal),
                    'path_removal_relative_error': causal, 'transport_passed': change <= .01,
                    'removal_passed': causal <= .01}
            saved[pop] = {**values, 'metadata': meta}; results[pop] = groups
    predictions = {'pred_a_instrument': live and all(a['passed'] for a in audits),
                   'pred_b_grouped_transport': all(g['transport_passed'] for p in results.values() for g in p.values()),
                   'pred_c_live_removal': all(g['removal_passed'] for p in results.values() for g in p.values())}
    torch.save(saved, rows)
    bound = ['CROSS_LAYER_ENDPOINT_TRANSPORT_V1_PREREGISTRATION.md', 'forward_endpoint_program_reference.py',
             'frozen_query_read_reference.py', 'query_hop_fork_reference.py', 'suffix_join_middle_match_reference.py',
             'exact_source_edit_reference.py', 'field_intervention_metrics.py']
    result = {'scope': 'Opened conditional cross-layer path screen; all native gates and coefficients retained.',
              'predictions': predictions, 'populations': results, 'controls': controls,
              'oracle_max_abs': max(a['max_abs'] for a in audits), 'independent_worlds': 32,
              'pairs': 384, 'executions': 768, 'opaque_export_constants': program.independent_constant_count(),
              'rows_sha256': digest(rows), 'runner_sha256': digest(Path(__file__)),
              'bound_sha256': hashlib.sha256(b''.join((POLY/n).read_bytes() for n in bound)).hexdigest(),
              'wall_seconds': time.perf_counter()-started}
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
