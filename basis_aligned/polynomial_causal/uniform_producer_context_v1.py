"""Uniform embedding-token law with actual upstream hierarchy scalar edits.

Unvalidated implementation until UNIFORM_PRODUCER_RANKING_V1 executes.
Native FP32 forward ports; FP64 conditional final mixed-operator evaluation.
"""
import importlib.util
import torch
import torch.nn.functional as F
from coupled_writer_tail_control_v1 import native_blocks
from coupled_writer_tail_v1 import Tail
from retained_objective_context_v1 import Contexts, P
from retained_contraction_error_control_v1 import components
from extracted_circuits.three_corner_head17_interaction_v1.ports import EPS, project, from_projections
from three_group_shared_dag_v1 import execute as compact


class UniformProducer(Contexts):
    def __init__(self):
        super().__init__()
        self.blocks = native_blocks(self.sd, 0, 10)
        self.tail = Tail(self.sd, torch.float32)
        self.tail64 = Tail(self.sd)
        spec = importlib.util.spec_from_file_location(
            'uniform_hierarchy', P/'extracted_circuits/crossfirst_state_executor_v1/hierarchy.py')
        self.hierarchy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hierarchy)
        self.hierarchy_weights = self.hierarchy.executor.load_weights(self.sd)
        self.upstream_writer = torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',
                                          weights_only=True)['direction'].float()

    @torch.no_grad()
    def sample(self, seed, batch=16, suffix_fp64=False, native_site=False, sequence_length=5):
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randint(50304, (batch, sequence_length), generator=generator)
        initial = F.rms_norm(self.sd['transformer.wte.weight'][ids].float(), (1152,))
        x, inherited = initial, None
        ports = {}
        for layer, block in self.blocks.items():
            raw = block.lambdas[0]*x + block.lambdas[1]*initial
            normalized = F.rms_norm(raw, (1152,))
            attention, inherited = block.attn(normalized, inherited)
            x = raw + attention
            if layer == 7:
                ports['mlp7'] = F.rms_norm(x, (1152,))
            if layer == 8:
                ports['attention8'] = normalized
                ports['rms8_squared'] = x.square().mean(-1).double()+EPS
            if layer == 9:
                fields = self.hierarchy.split_fields(
                    ids, ports['mlp7'], ports['attention8'], normalized,
                    ports['rms8_squared'], (raw.square().mean(-1)+EPS).sqrt().double(),
                    self.hierarchy_weights)
                break
            x = x + block.mlp(F.rms_norm(x, (1152,)))
        edits = [(fields[name][..., None]*self.upstream_writer).float()
                 for name in ('child', 'remainder')]
        corners = [x] + ([raw+(attention-edit) for edit in edits] if native_site
                         else [x-edit for edit in edits])
        if suffix_fp64:
            raw17 = [self.tail64(c.double(), initial.double(), inherited.double()) for c in corners]
        else:
            raw17 = [self.tail(c, initial, inherited).double() for c in corners]
        first = inherited.reshape(batch, sequence_length, 9, 128)[:, :, 2].double()
        attention_ports = [
            from_projections(project(r, self.maps), r.square().mean(-1)+EPS, first, self.mixture)
            for r in raw17]
        additive = raw17[1]+raw17[2]-raw17[0]
        attention_ports.append(from_projections(
            project(additive, self.maps), additive.square().mean(-1)+EPS, first, self.mixture))
        terms = components(*attention_ports)
        exact_compact = compact(*attention_ports)
        producer_replay = float((terms.sum(1)-exact_compact).norm()/exact_compact.norm().clamp_min(1e-30))
        pre_mlp = [r+self.tail64.attention(r, inherited.double(), 17) for r in raw17]
        z = (pre_mlp[1]+pre_mlp[2]-pre_mlp[0])[:, -1]
        state = z+terms.sum(1)@self.writer.T
        r2 = state.square().mean(-1)+EPS
        normalized = state/r2.sqrt()[:, None]
        final = state+((normalized@self.left.T)*(normalized@self.right.T))@self.down.T+self.bias
        denominator = r2*(final.square().mean(-1)+EPS).sqrt()
        diagnostics = {
            'producer_replay': producer_replay,
            'child_amplitude_rms': float(fields['child'].square().mean().sqrt()),
            'remainder_amplitude_rms': float(fields['remainder'].square().mean().sqrt()),
            'background_rms': float(z.square().mean().sqrt()),
            'retained_write_rms': float(terms.sum(1).square().mean().sqrt()),
            'finite': all(bool(torch.isfinite(v).all()) for v in (z, terms, denominator)),
        }
        return z, terms, denominator, diagnostics
