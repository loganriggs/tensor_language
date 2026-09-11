"""Lift a frozen reader to the complete native quadratic weight tensor.

A polarization/removal and metric-SVD identities <=1e-9.
B old node versus native projected function cosine >=.8.
C old two-writer span covers >=50% of native projected coefficient energy.
No reader fitting, corpus use, or native behavioral claim.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from shared_input_factor_v1 import native_partner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root = Path(__file__).parent
    output = root/'SHARED_PARENT1_NATIVE_LIFT_V1.json'
    artifact = root/'SHARED_PARENT1_NATIVE_LIFT_V1.pt'
    assert not output.exists() and not artifact.exists()
    source = root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    old = torch.load(source, weights_only=True, map_location='cpu')
    node = old['nodes'][1]; u = node['reader']; u = u/u.norm()
    binding = json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(Path(p) for p in binding if p.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    left, right, down = [weights[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    writers = old['output_whitener']@down
    m = native_partner(u, left, right, writers)
    frozen = node['writers']@node['partners'].T
    def transform(matrix):
        return matrix+(math.sqrt(2)-1)*(matrix@u)[:, None]*u[None]
    n, previous = transform(m), transform(frozen)
    energy = .5*n.square().sum()
    cosine = float((n*previous).sum()/n.norm()/previous.norm())
    old_basis = torch.linalg.qr(node['writers'], mode='reduced').Q
    span_capture = float((old_basis.T@n).square().sum()/n.square().sum())
    left_vectors, singular, right_vectors = torch.linalg.svd(n, full_matrices=False)
    cumulative = singular.square().cumsum(0)/singular.square().sum()
    rank95 = int((cumulative < .95).sum())+1
    rank99 = int((cumulative < .99).sum())+1
    torch.manual_seed(4401); x = torch.randn(11, len(u))
    removed = x-(x@u)[:, None]*u[None]
    def native(v): return ((v@left.T)*(v@right.T))@writers.T
    difference = native(x)-native(removed)
    predicted = (x@u)[:, None]*(x@m.T)
    replay = float((difference-predicted).norm()/difference.norm())
    norm_error = float(abs(.5*singular.square().sum()-energy)/energy)
    rank = 16
    partners = right_vectors[:rank].T
    partners += (2**-.5-1)*u[:, None]*(u@partners)[None]
    output_writers = left_vectors[:, :rank]*singular[:rank]
    torch.save(dict(reader=u, writers=output_writers, partners=partners,
        output_whitener=old['output_whitener'], scope='Top16 native projection branches at frozen parent1 reader; no data fitting.'), artifact)
    result = dict(pred_a=max(replay,norm_error)<=1e-9, pred_b=cosine>=.8, pred_c=span_capture>=.5,
        execution_replay=replay, svd_norm_error=norm_error, function_cosine=cosine,
        old_writer_span_capture=span_capture,
        old_node_over_native_projection_energy=float(previous.square().sum()/n.square().sum()),
        full_native_projection_coefficient_fraction=float(energy/99245061353.47293),
        rank95=rank95, rank99=rank99,
        rank_curve={str(r):float(cumulative[r-1]) for r in [1,2,4,8,16,32,64,128,256,512,1152]},
        top_native_output_projection_into_old_span=(old_basis.T@left_vectors[:, :16]).square().sum(0).tolist(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint=dict(path=str(checkpoint), bound_sha256=binding[str(checkpoint)]),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        scope='Complete native orthogonal shared-factor projection at one frozen reader, '
              'not the graph-node removal or input intervention through RMS. No behavioral/circuit claim.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
