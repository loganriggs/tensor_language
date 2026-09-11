"""Exact global shared-reader rewrite, with no rank truncation or fitting.

A original/rebased executor, thin coefficient, and input/node-removal <=1e-9.
B both registered common-reader node pairs have cross-start cosine >=.8.
C all four nodes have at least two consumers above1% own-group removal energy.
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v2 import factors, execute, price
from shared_parent_intervention_v1 import banks, disable, value
from shared_input_factor_v1 import native_partner


@torch.no_grad()
def rebase(original, reader):
    a, s, c = factors(original)
    u = reader/reader.norm()
    groups = []
    for aa, ss, cc in zip(a, s, c):
        residual = aa-(aa@u)[:, None]*u[None]
        private = torch.linalg.qr(residual.T, mode='reduced').Q.T
        ap = aa@private.T
        core = (ap.T*ss)@ap
        lam, rotation = torch.linalg.eigh(core)
        private = rotation.T@private
        au, av = aa@u, aa@private.T
        groups.append(dict(parent_ids=torch.tensor([0]), pair_ids=torch.tensor([0]),
            shared_coeff=((au.square()*ss).sum()).reshape(1),
            private=private, lam=lam, cross=(2*((au*ss)@av)).reshape(1,-1), writer=cc))
    return dict(readers=u[None], pairs=torch.tensor([[0,0]]), groups=groups,
                output_whitener=original['output_whitener'], bias=original['bias'])


@torch.no_grad()
def inspect(original, graph):
    old = factors(original); new = factors(graph)
    u = graph['readers'][0]
    checks = []
    for aa, ss, bb, tt in zip(old[0],old[1],new[0],new[1]):
        joined = torch.cat((aa,bb))
        qr = torch.linalg.qr(joined.T, mode='reduced').R
        delta = (qr*torch.cat((ss,-tt)))@qr.T
        old_norm = ((aa@aa.T).square()*ss[:,None]*ss[None]).sum().sqrt()
        checks.append(float(delta.norm()/old_norm))
    torch.manual_seed(4501); x = torch.randn(11, len(u))
    before, after = execute(original,x), execute(graph,x)
    checks.append(float((before-after).norm()/before.norm()))
    node = banks(graph)[2][0]
    node_removed = after-disable(graph,x,[0])
    projected_input = x-(x@u)[:,None]*u[None]
    global_removed = after-execute(graph,projected_input)
    checks.extend([float((node_removed-global_removed).norm()/global_removed.norm()),
                   float((node_removed-value(node,x)).norm()/node_removed.norm())])
    a,s,c = old
    qu = torch.einsum('grd,gr->gd',a,s*(a@u))
    quadratic_energy = ((a@a.transpose(1,2)).square()*s[:,:,None]*s[:,None]).sum((1,2))
    fractions = (2*qu.square().sum(1)-(qu@u).square())/quadratic_energy
    # Compact complete node: all shared-factor terms of the original graph.
    partners = 2*qu-(qu@u)[:,None]*u[None]
    matrix = c.T@partners
    return dict(maximum_replay=max(checks), effective_consumers=int((fractions>=.01).sum()),
        group_shared_factor_fractions=fractions.tolist(), price=price(graph,old)), matrix


def matrix_inner(u,m,v,n):
    return .5*((u@v)*(m*n).sum()+(m@v)@(n@u))


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root=Path(__file__).parent
    output=root/'GLOBAL_READER_REBASE_V1_RESULT.json'
    assert not output.exists()
    sources={}
    originals=[]
    for label in ('SPECTRAL','NATIVE'):
        path=root/f'SHARED_READER_JOINT_FIT_V1_{label}_GRAPH.pt'
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        originals.append(torch.load(path,weights_only=True,map_location='cpu'))
    rows=[]
    for parent in (1,9):
        u=originals[0]['readers'][parent];u=u/u.norm()
        pairs=[inspect(old,rebase(old,u)) for old in originals]
        a,b=[item[1] for item in pairs]
        aa,bb,ab=matrix_inner(u,a,u,a),matrix_inner(u,b,u,b),matrix_inner(u,a,u,b)
        cosine=float(ab/(aa*bb).sqrt())
        rows.append(dict(spectral_reader=parent,scope_results=[item[0] for item in pairs],
            common_interface_function_cosine=cosine,
            relative_function_difference=float(((aa+bb-2*ab)/aa).clamp_min(0).sqrt())))
    result=dict(pred_a=all(s['maximum_replay']<=1e-9 for r in rows for s in r['scope_results']),
        pred_b=all(r['common_interface_function_cosine']>=.8 for r in rows),
        pred_c=all(s['effective_consumers']>=2 for r in rows for s in r['scope_results']),
        rows=rows,sources=sources,
        scope='Exact change of graph consumers at fixed common readers; no approximation or fitting. '
              'Extra storage/arithmetic charged. Common-interface agreement is not semantic identification.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for row in rows:
        print(json.dumps(dict(parent=row['spectral_reader'],cosine=row['common_interface_function_cosine'],
            effective_consumers=[s['effective_consumers'] for s in row['scope_results']],
            maximum_replay=max(s['maximum_replay'] for s in row['scope_results']),
            price=row['scope_results'][0]['price']),indent=2))


if __name__=='__main__':main()
