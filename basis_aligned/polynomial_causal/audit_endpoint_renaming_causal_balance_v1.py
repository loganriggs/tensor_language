"""Saved physical-removal accounting in the fixed semantic label frame."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch
import entity_equivariance_reference as J
import field_intervention_metrics as M

BASE = Path(__file__).resolve().parent
SOURCE = BASE/'CROSS_LAYER_ENDPOINT_TRANSPORT_V1_ROWS.pt'
OUT = BASE/'ENDPOINT_RENAMING_CAUSAL_BALANCE_V1_RESULT.json'
def center(x): return x-x.mean(-1, keepdim=True)
def rms(x): return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    signal.alarm(180); torch.set_num_threads(2); started = time.perf_counter()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_hash == '50374731a2cbdb44169c2c3c67dceac958235548eb5dbf497206f608029c7498'
    assert not OUT.exists(); controls = J.controls(); assert controls['passed']
    data = torch.load(SOURCE, map_location='cpu', weights_only=True); results = {}; audits = []
    for pop, block in data.items():
        metadata = block['metadata']; old = [i for i,m in enumerate(metadata) if m['case'] == 'base']
        new = [i for i,m in enumerate(metadata) if m['case'] == 'both']; maps = []
        for i,j in zip(old,new):
            a,b = metadata[i],metadata[j]
            assert all(a[k] == b[k] for k in ('world','order','orientation','hop','middle','foil'))
            pi = torch.arange(29); pi[a['middle']],pi[a['foil']] = a['foil'],a['middle']
            assert int(pi[a['answer']]) == b['answer']; maps.append(pi)
        assert len(old) == len(new) == 192; pi = torch.stack(maps)
        native = block['native_query_logits']; removal = block['removal']; cut = native-removal
        delta = {k:center(v[new].gather(1,pi)-v[old]) for k,v in (('native',native),('removal',removal),('cut',cut))}
        audits.append(M.correspondence(delta['native'],delta['removal']+delta['cut']))
        radius,_ = J.information_radius(native[old],native[new].gather(1,pi)); radius = radius.clamp_min(0)
        groups = {}
        for hop in range(4):
            sel = torch.tensor([metadata[i]['hop'] == hop for i in old]); l,r,b = (delta[k][sel] for k in ('native','removal','cut'))
            den = max(float(l.square().sum()),1e-12); base_rms = rms(center(removal[old][sel]))
            rel = rms(r)/max(base_rms,1e-6); rb_cos = float((r*b).sum())/max(float(r.norm()*b.norm()),1e-12)
            js = float(radius[sel].mean())
            groups[str(hop)] = {'pairs':int(sel.sum()),'native_delta_rms':rms(l),'removal_delta_rms':rms(r),
                'cut_delta_rms':rms(b),'base_removal_rms':base_rms,'removal_relative_change':rel,
                'removal_cut_cosine':rb_cos,'native_over_removal_delta_rms':rms(l)/max(rms(r),1e-6),
                'removal_projection_on_native_delta':float((r*l).sum())/den,
                'cut_projection_on_native_delta':float((b*l).sum())/den,
                'floors':{'removal_base':base_rms<1e-6,'removal_delta':rms(r)<1e-6,'projection':float(l.square().sum())<1e-12},
                'minimum_paired_mean_query_kl':js,'invariant_output_ruled_out_at_query_1e3':js>1e-3,
                'causal_field_transport_passed':rel<=.01,
                'compensating_remainder_diagnostic':rb_cos<0 and rms(l)<rms(r)}
        results[pop] = groups
    result = {'scope':'Opened fixed-label physical-removal accounting, not a new replacement or filtered cohort.',
        'pred_a_instrument':all(a['passed'] for a in audits),'pred_b_invariant_causal_field':all(g['causal_field_transport_passed'] for p in results.values() for g in p.values()),
        'oracle_max_abs':max(a['max_abs'] for a in audits),'controls':controls,'populations':results,
        'source_sha256':source_hash,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prereg_sha256':hashlib.sha256((BASE/'ENDPOINT_RENAMING_CAUSAL_BALANCE_V1_PREREGISTRATION.md').read_bytes()).hexdigest(),
        'independent_worlds':32,'pairs':384,'wall_seconds':time.perf_counter()-started}
    assert all(torch.isfinite(v).all() for v in delta.values())
    OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
