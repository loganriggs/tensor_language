"""Frozen no-fit pair-frame diagnostic, board11:00UTC; reused development panels."""
import json
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts, P
from shared_local_fineweb_v1 import digest


@torch.no_grad()
def main():
    out = P/'PAIR_ERROR_DECOMPOSITION_V1_RESULT.json'
    assert not out.exists()
    start = time.monotonic(); torch.set_num_threads(2)
    model = Contexts(); panels = []
    for cache, rowfile, offset in [('COMPOSED_LAST_BLOCK_STATES_V1', None, 0),
                                   ('MINIMAX_FRESH_CACHE_V1', 'MINIMAX_FRESH_CACHE_V1_ROWS', 24)]:
        path = P/(cache+'_ARTIFACT.pt')
        assert digest(path) == json.loads((P/(cache+'_RESULT.json')).read_text())['artifact_sha']
        data = torch.load(path, weights_only=True)
        if rowfile:
            rows = json.loads((P/(rowfile+'.json')).read_text())['rows']
        else:
            rows = json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24] + json.loads((P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json').read_text())['rows']
        z = data['linear_parts'][:,3].double()[offset:]
        state = data['linear_parts'][:,2].double()[offset:]
        a = torch.linalg.lstsq(model.writer, (state-z).T).solution.T
        eps = torch.finfo(torch.float32).eps
        inv = 1/((state.square().mean(-1)+eps)*(data['states'][offset:,2].double().square().mean(-1)+eps).sqrt())
        lookup = {v:k for k,v in enumerate(model.ids)}
        rows = rows[offset:]
        uk = torch.tensor([lookup[r['uk_id']] for r in rows]); us = torch.tensor([lookup[r['us_id']] for r in rows]); idx = torch.arange(len(rows))
        scores = []
        for name in ['SPARSE_INTERACTION_EXECUTOR_V1', 'PAIR_FRAME_SPARSE_V1']:
            error = model.decode(name)-model.tensor
            def contract(x,y): return torch.einsum('oih,ni,nh->no', error, x, y)
            raw = contract(z,a); value = raw*inv[:,None]
            dz = z[::2]-z[1::2]; da = a[::2]-a[1::2]
            zm = (z[::2]+z[1::2])/2; am = (a[::2]+a[1::2])/2
            im = (inv[::2]+inv[1::2])/2
            terms = torch.stack([contract(dz,am)*im[:,None], contract(zm,da)*im[:,None],
                                 (raw[::2]+raw[1::2])/2*(inv[::2]-inv[1::2])[:,None]],1)
            delta = value[::2]-value[1::2]
            replay = float((terms.sum(1)-delta).norm()/delta.norm())
            selected = value[idx,uk]-value[idx,us]
            groups = []
            for begin in range(0,len(rows),24):
                v = value[begin:begin+24]
                groups.append(dict(group=begin//24, all6_energy=float((v[:,::2]-v[:,1::2]).square().sum()/2),
                                   selected_energy=float(selected[begin:begin+24].square().sum()),
                                   endpoint_selected_energies=[dict(uk_id=k, energy=float(selected[torch.tensor([j for j in range(begin,begin+24) if rows[j]['uk_id']==k])].square().sum())) for k in sorted({r['uk_id'] for r in rows[begin:begin+24]})]))
            scores.append(dict(name=name, all12_energy=float(value.square().sum()),
                               all6_energy=float((value[:,::2]-value[:,1::2]).square().sum()/2),
                               selected_energy=float(selected.square().sum()), groups=groups,
                               cue_error_terms=terms.tolist(), replay_relative=replay,
                               term_order=['background','edited_head','supplied_denominator']))
        ratios = {k:scores[1][k]/scores[0][k] for k in ['all12_energy','all6_energy','selected_energy']}
        panels.append(dict(cache=cache, scores=scores, pair_over_original=ratios))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), panels=panels,
                  pred_a=all(s['replay_relative']<=1e-10 for p in panels for s in p['scores']),
                  pred_b=all(p['pair_over_original']['all6_energy']<=.95 for p in panels),
                  seconds=time.monotonic()-start,
                  scope='No fit or new forwards; reused development panels. Signed cue-error decomposition, not independent energy attribution; supplied normalization remains external.')
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='panels'}))
    print(json.dumps([p['pair_over_original'] for p in panels]))


if __name__=='__main__': main()
