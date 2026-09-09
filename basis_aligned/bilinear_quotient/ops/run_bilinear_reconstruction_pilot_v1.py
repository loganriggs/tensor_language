#!/usr/bin/env python3
"""Bounded reconstruction pilot. GPU execution only via managed enqueue.sh.

pred_a: complete tiny module/logit FP64 closure at atol=rtol=1e-9, exact rational identity.
pred_b: rational planted recovery, perturbation rejection and independent-router discrimination.
pred_c: unchanged contextual checkpoint execution and registered intervention fidelity.
pred_d: a previously unspecified reusable operation with held-out causal/extraction evidence.
Null: a faithful compiler and/or compact state description without a discovered causal operation.
Price: B<=8, feature banks<=256, tensors<=256 MiB; no training, 1800 s execution watchdog.
"""
# BQGATE: EXPERIMENT pred_a_faithful_execution pred_b_sound_discovery_controls pred_c_contextual_execution pred_d_reusable_computation_discovered
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
OUT = POLY / "BILINEAR_RECONSTRUCTION_PILOT_V1_RESULT.json"
PREREG = POLY / "BILINEAR_RECONSTRUCTION_PILOT_PREREGISTRATION.md"
REFERENCE = POLY / "bilinear_reconstruction_reference.py"
CHECKPOINT = ROOT / "runs_hop/attn-mlp-attn-rms-seed0/model.pt"
EXPECTED_REFERENCE = 'cb6617faa3ecaeb58d0e7e59570db0a5b5d0f16a04987fe3931dadb89d6839d8'
EXPECTED_PREREG = '1d0e501b394a186d3e1eaeaa660e1a2535d3b17409b8721f9db598aaf95456ed'


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(message):
    print(message, flush=True)


def phase_a(R, torch):
    records = []
    arms = {
        "native": [],
        "past_key_remove": [dict(layer=0, site="k1", position=2, head=0, kind="zero")],
        "value_interchange": [dict(layer=1, site="v", position=1, head=1, kind="swap")],
        "head_remove": [dict(layer=0, site="head", position=2, head=0, kind="zero")],
        "mlp_remove": [dict(layer=1, site="mlp", position=3, kind="zero")],
        "joint": [dict(layer=0, site="k1", position=2, head=0, kind="zero"),
                  dict(layer=1, site="mlp", position=3, kind="zero")],
    }
    for seed in (909, 910):
        for norm, rope in ((False, False), (True, False), (False, True), (True, True)):
            model = R.Tiny(seed, norm, rope)
            for length in (8, 16, 32):
                g = torch.Generator().manual_seed(seed+length)
                tokens = torch.randint(8, (2, length), generator=g)
                baseline, _ = model(tokens)
                _, donor = model((tokens+1)%8)
                for name, edits in arms.items():
                    direct, trace = model(tokens, edits=edits, donor=donor)
                    for backend in ("recurrent", "direct_step", "recurrent_step"):
                        actual, atrace = (model(tokens, backend, edits, donor) if backend == "recurrent"
                                          else model.stream(tokens, backend, edits, donor))
                        modules = {str(key): R.close(atrace[key], expected) for key, expected in trace.items()}
                        records.append(dict(seed=seed, rmsnorm=norm, rope=rope, length=length,
                                            arm=name, backend=backend, logits=R.close(actual, direct),
                                            module_max_abs=max(m["max_abs"] for m in modules.values()),
                                            modules_passed=all(m["passed"] for m in modules.values()),
                                            edit_effect_max_abs=float((direct-baseline).abs().max())))
                embeddings = torch.randn(2, length, 16, generator=g, dtype=torch.float64)*0.3
                direct, _ = model(tokens, embeddings=embeddings)
                recurrent, _ = model(tokens, "recurrent", embeddings=embeddings)
                records.append(dict(seed=seed, rmsnorm=norm, rope=rope, length=length,
                                    arm="continuous_embeddings", backend="recurrent",
                                    logits=R.close(recurrent, direct), modules_passed=True))
    rational = R.rational_attention_check()
    passed = rational["passed"] and all(r["logits"]["passed"] and r["modules_passed"] for r in records)
    live = all(r.get("edit_effect_max_abs", 1) > 1e-12 for r in records if r["arm"] not in ("native", "continuous_embeddings"))
    return dict(passed=passed and live, live_edits=live, rational=rational, records=records,
                max_logit_abs=max(r["logits"]["max_abs"] for r in records),
                max_module_abs=max(r.get("module_max_abs", 0) for r in records),
                domain="complete tiny model; independent real input/weight fixture and token interventions")


def tensor_matrix(matrix, torch):
    return torch.tensor([[float(v) for v in row] for row in matrix.tolist()], dtype=torch.float64)


def local_bank(factors, torch):
    # Two head memories, 4x4x4 each. No global state/Gram construction.
    k1, k2, v = (factors[name][..., :2, :4] for name in ("k1", "k2", "v"))
    shape = (*k1.shape[:2], 2, 4, 4, 4)
    R = sys.modules["bilinear_reconstruction_reference"]
    R.check_shape(shape, k1.dtype)
    return torch.einsum("bthi,bthj,bthv->bthijv", k1, k2, v).flatten(2).reshape(-1, 128)


def phase_b(R, torch):
    import sympy as s
    records, positive_lift, positive_rows = {}, None, None
    for kind in ("planted", "perturbed", "independent_routers"):
        started = time.perf_counter()
        f = R.exact_fixture(kind)
        lift, update, rows, columns = R.exact_span(f["update"])
        joined = f["head_updates"][0].row_join(f["head_updates"][1])
        ilift, iupdate, irows, icolumns = R.exact_span(joined)
        # Reader observability is assessed on the intervention-closed state, not assumed.
        folded = f["readers"]*ilift
        observable_rank = folded.rank()
        same_positive_span = True if positive_lift is None else positive_lift*f["update"][positive_rows, :] == f["update"]
        if kind == "planted":
            positive_lift, positive_rows = lift, rows
        g = torch.Generator().manual_seed(1919)
        max_error = 0.0
        # Two independent contextual input/query streams, unseen combinations and lengths.
        full_U = [tensor_matrix(u, torch) for u in f["head_updates"]]
        red_U = [tensor_matrix(u[irows, :], torch) for u in f["head_updates"]]
        full_W, red_W = tensor_matrix(f["readers"], torch), tensor_matrix(folded, torch)
        for length in (8, 16, 32):
            source = torch.randn(length, 4, generator=g, dtype=torch.float64)*0.2
            donor = torch.randn(length, 4, generator=g, dtype=torch.float64)*0.2
            query = torch.randn(length, 4, generator=g, dtype=torch.float64)*0.2
            for arm in ("native", "remove0", "swap1", "joint"):
                full_state, reduced_state = torch.zeros(128, dtype=torch.float64), torch.zeros(ilift.cols, dtype=torch.float64)
                for t in range(length):
                    for h in range(2):
                        if t == 2 and h == 0 and arm in ("remove0", "joint"):
                            continue
                        x = donor[t] if t == 1 and h == 1 and arm in ("swap1", "joint") else source[t]
                        phi = torch.stack([x[a]*x[b]*x[c] for a, b, c in f["cubic"]])
                        full_state += full_U[h] @ phi
                        reduced_state += red_U[h] @ phi
                    psi = torch.stack([query[t,a]*query[t,b] for a,b in f["quadratic"]])
                    native = (full_W@full_state).reshape(8,10)@psi
                    compiled = (red_W@reduced_state).reshape(8,10)@psi
                    error = R.close(compiled, native)
                    if not error["passed"]:
                        raise ArithmeticError(f"{kind} exact-program numeric execution failed: {error}")
                    max_error = max(max_error, error["max_abs"])
        active = sum(any(f["update"][:,j]) for j in range(20))
        record = dict(natural_rank=lift.cols, intervention_closed_rank=ilift.cols,
                      observable_intervention_rank=observable_rank,
                      positive_span_still_exact=same_positive_span,
                      exact_factorization=True, router_functions_identical=f["router_functions_identical"],
                      active_cubic_features=active, numeric_max_abs=max_error,
                      natural_state_scalars=128, discovered_intervention_state_scalars=ilift.cols,
                      elementary_per_head_cubic_state_scalars=2*active,
                      seconds=time.perf_counter()-started,
                      interpretation="known polynomial-feature sharing control; no learned semantic claim")
        if kind == "planted":
            record["program"] = {
                "source_monomials": f["cubic"], "query_monomials": f["quadratic"],
                "encoder_selected_state_rows": irows,
                "equations": "z += U0 phi(x0) + U1 phi(x1); y=reshape(Wz,8,10) psi(q). Skip or swap only the registered head/source update.",
                "update0": [[str(v) for v in row] for row in f["head_updates"][0][irows,:].tolist()],
                "update1": [[str(v) for v in row] for row in f["head_updates"][1][irows,:].tolist()],
                "folded_readers": [[str(v) for v in row] for row in folded.tolist()],
                "lift_certificate": [[str(v) for v in row] for row in ilift.tolist()],
                "runtime_uses_lift": False,
                "dense_fixture_basis_adapter_scalars": 128*128,
                "structural_gain_over_elementary_cubic_baseline": False,
            }
        records[kind] = record
        log(f"phase B {kind}: natural rank {lift.cols}, edit-closed rank {ilift.cols}, {record['seconds']:.2f}s")
    # Known token-count construction, independent from contextual polynomial fixtures.
    g = torch.Generator().manual_seed(2909)
    table = torch.randn(128,8,generator=g,dtype=torch.float64)
    ids = torch.randint(8,(32,),generator=g)
    counts = torch.bincount(ids,minlength=8).double()
    count_error = R.close(table[:,ids].sum(1),table@counts)
    # Generic, fully contextual Tiny model, so polynomial-count controls cannot stand in for it.
    tiny = R.Tiny(seed=910)
    tokens = torch.randint(8,(8,32),generator=g)
    _, trace = tiny(tokens)
    features = local_bank({name:trace[(1,name)] for name in ('k1','k2','v')},torch)
    singular = torch.linalg.svdvals(features)
    rank = int((singular > singular[0]*1e-10).sum())
    passed = (records['planted']['natural_rank'] == 4
              and records['planted']['intervention_closed_rank'] > 4
              and not records['perturbed']['positive_span_still_exact']
              and records['perturbed']['natural_rank'] > 4
              and records['independent_routers']['natural_rank'] > 4
              and not records['independent_routers']['router_functions_identical']
              and count_error['passed'])
    return dict(passed=passed, fixtures=records, count_baseline=count_error,
                count_baseline_state=8, count_baseline_constants=128*8,
                generic_contextual=dict(numerical_rank=rank, bank_columns=128,
                    relative_smallest_singular=float(singular[-1]/singular[0]),
                    scope='finite-sample diagnostic, not exact algebraic rank or semantic discovery'))


def distribution_error(actual, teacher, torch):
    a, b = actual.double().log_softmax(-1), teacher.double().log_softmax(-1)
    kl = (b.exp()*(b-a)).sum(-1).clamp_min(0).flatten()
    return dict(mean_kl=float(kl.mean()), p99_kl=float(torch.quantile(kl,.99)), max_kl=float(kl.max()))


def common_factor_screen(model):
    from fractions import Fraction
    def canonical(row):
        vals = [Fraction.from_float(float(x)) for x in row]
        pivot = next((x for x in vals if x), None)
        return tuple(x/pivot for x in vals) if pivot else tuple(vals)
    mlp = model.layers[1]
    left = [canonical(row) for row in mlp.L.weight.detach().cpu()]
    right = [canonical(row) for row in mlp.R.weight.detach().cpu()]
    products = [tuple(sorted((a,b))) for a,b in zip(left,right)]
    return dict(native_products=len(products), distinct_unordered_products=len(set(products)),
                total_factor_occurrences=len(left)+len(right), distinct_proportional_factors=len(set(left+right)),
                arithmetic='exact Fraction canonicalization of stored float weights',
                scope='proportional linear factors and duplicate unordered products only; no claim against other nonlinear decompositions')


def phase_c(R, torch):
    from hop_ablate import load
    from hop_data import sample_docs
    device = 'cuda'
    if not torch.cuda.is_available():
        return dict(status='not_run_no_gpu',passed=False)
    model, config = load('attn-mlp-attn-rms-seed0')
    weights_before = digest(CHECKPOINT)
    model = model.to(device=device,dtype=torch.float64).eval()
    model.requires_grad_(False)
    factor_screen = common_factor_screen(model)
    def docs(seed,length):
        return sample_docs(8,torch.Generator().manual_seed(seed))[0][:,:length].to(device)
    discovery = docs(1909,64)
    teacher = model(discovery)
    direct, trace = R.small_forward(model,discovery,capture=True)
    native_closure = R.close(direct,teacher)
    if not native_closure['passed']:
        return dict(status='invalid_native_adapter',passed=False,native_closure=native_closure)
    features = local_bank({name:trace[(2,name)] for name in ('k1','k2','v')},torch)
    _, singular, vt = torch.linalg.svd(features,full_matrices=False)
    rank = int((singular>singular[0]*1e-10).sum())
    basis = vt[:rank].T.contiguous()
    # Freeze here before either independent population is constructed.
    basis_sha = hashlib.sha256(basis.cpu().numpy().tobytes()).hexdigest()
    log(f'phase C frozen local update rank {rank}/128; beginning held-out validation')
    records = []
    arms = {
        'native':[],
        'upstream_head_remove':[dict(layer=0,site='head',position=3,head=0,kind='zero')],
        'past_key_remove':[dict(layer=2,site='k1',position=2,head=0,kind='zero')],
        'value_interchange':[dict(layer=2,site='v',position=5,head=1,kind='swap')],
        'joint':[dict(layer=0,site='head',position=3,head=0,kind='zero'),
                 dict(layer=2,site='v',position=5,head=1,kind='swap')],
    }
    for label,seed,length in (('IID',1910,64),('longer_prefix',1911,128)):
        tokens = docs(seed,length)
        base,_ = R.small_forward(model,tokens)
        _,donor = R.small_forward(model,tokens.roll(1,0),capture=True)
        for arm,edits in arms.items():
            direct,dtrace = R.small_forward(model,tokens,edits=edits,donor=donor,capture=True)
            recurrent,_ = R.small_forward(model,tokens,'recurrent',edits,donor)
            bank = local_bank({name:dtrace[(2,name)] for name in ('k1','k2','v')},torch)
            span_error = R.errors((bank@basis)@basis.T,bank)
            native_effect = direct-base
            rec_base,_ = R.small_forward(model,tokens,'recurrent') if arm == 'native' else (base,None)
            # End-to-end output closure already tests edited trajectories. For effect vectors,
            # use the recurrent native output, not the direct native output as an anchor.
            if arm == 'native':
                recurrent_baseline = rec_base
            effect = recurrent-recurrent_baseline
            native_centered = native_effect-native_effect.mean(-1,keepdim=True)
            centered = effect-effect.mean(-1,keepdim=True)
            record = dict(population=label,seed=seed,length=length,arm=arm,
                          logits=R.close(recurrent,direct),distribution=distribution_error(recurrent,direct,torch),
                          effect=R.errors(centered,native_centered),local_span_error=span_error,
                          effect_norm=float(native_centered.norm()),near_zero_floor=1e-8)
            record['effect_passed'] = (record['effect']['max_abs']<=1e-9 if record['effect_norm']<=1e-8
                                       else record['effect']['relative_l2']<=.01)
            records.append(record)
    # Practical full-model prefill and complete cached decode at identical dtype and weights.
    tokens = discovery[:1]
    benchmarks = {backend:R.benchmark(lambda:R.small_forward(model,tokens,backend),device)
                  for backend in ('direct','recurrent')}
    decode_closure = {}
    for backend in ('direct_step','recurrent_step'):
        cache,outputs = (backend,{}),[]
        for t in range(tokens.shape[1]-1):
            y,cache = R.small_decode_step(model,tokens[:,t],t,cache)
            outputs.append(y)
        benchmark_cache = cache
        benchmarks[backend] = R.benchmark(
            lambda:R.small_decode_step(model,tokens[:,-1],tokens.shape[1]-1,benchmark_cache),device)
        y,_ = R.small_decode_step(model,tokens[:,-1],tokens.shape[1]-1,cache)
        outputs.append(y)
        decode_closure[backend] = R.close(torch.stack(outputs,1),model(tokens))
    # Deployed FP32 comparison is separate from the FP64 identity test.
    model = model.float()
    direct32,_ = R.small_forward(model,tokens)
    recurrent32,_ = R.small_forward(model,tokens,'recurrent')
    fp32 = dict(logits=R.close(recurrent32,direct32,atol=1e-4,rtol=1e-4),
                distribution=distribution_error(recurrent32,direct32,torch))
    constants = sum(p.numel() for p in model.parameters())
    linear_macs_per_token = 2*6*128**2 + 3*128*512 + 128*29
    complete_costs = {}
    for length in (64,128,240):
        price = R.attention_cost(1,2,4,length,32,bytes_per_scalar=8)
        price.update(stored_weight_scalars=constants,stored_weight_bytes_fp64=constants*8,
                     unchanged_linear_macs_per_token=linear_macs_per_token,
                     direct_total_prefill_macs=length*linear_macs_per_token+price['direct_prefill_triangular_macs'],
                     recurrent_total_prefill_macs=length*linear_macs_per_token+price['recurrent_prefill_macs'],
                     direct_total_decode_macs=linear_macs_per_token+price['direct_decode_macs'],
                     recurrent_total_decode_macs=linear_macs_per_token+price['recurrent_decode_macs'],
                     lower_order_excluded='norms, rotary, elementwise products/adds, embedding lookup, workspace')
        complete_costs[str(length)] = price
    passed = (native_closure['passed'] and all(r['logits']['passed'] and r['effect_passed'] for r in records)
              and all(r['passed'] for r in decode_closure.values()) and weights_before==digest(CHECKPOINT))
    sharing_candidate = rank<128 or factor_screen['distinct_unordered_products']<factor_screen['native_products']
    return dict(status='local_sharing_candidate_requires_followup' if sharing_candidate else 'no_local_shared_operation',
                passed=passed,checkpoint_sha256=weights_before,config=config,native_closure=native_closure,
                local_rank=rank,relative_smallest_singular=float(singular[-1]/singular[0]),
                frozen_basis_sha256=basis_sha,common_factors=factor_screen,records=records,
                decode_closure=decode_closure,fp32=fp32,benchmarks=benchmarks,
                model_constant_scalars=constants,
                complete_costs=complete_costs,
                extraction_scope='full unchanged small-model token-to-logit execution; discovered reduced circuit not established',
                reduced_program_heldout_causal_validation='not reached; no validated structurally simpler candidate',
                iid_scope='fresh generated documents, training-set membership unknown',
                ood_scope='longer prefixes than discovery, within training context; not a claim of unseen task families')


def main():
    if digest(REFERENCE)!=EXPECTED_REFERENCE or digest(PREREG)!=EXPECTED_PREREG:
        raise RuntimeError('reviewed reference or preregistration changed')
    dryrun = os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1'
    if dryrun:
        log(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,queue_touched=False,
                            phases=['A','B','conditional C'],feature_bank_max=128,tensor_cap_bytes=256*2**20,
                            watchdog_seconds=1800,checkpoint_exists=CHECKPOINT.is_file(),
                            reference_exists=REFERENCE.is_file(),prereg_exists=PREREG.is_file())))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    def timeout_handler(_signum,_frame):
        raise TimeoutError('pilot exceeded registered 1800-second execution watchdog')
    signal.signal(signal.SIGALRM,timeout_handler)
    signal.alarm(1800)
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(POLY))
    import torch
    import bilinear_reconstruction_reference as R
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started,time_cpu = time.perf_counter(),time.process_time()
    result = dict(schema='bilinear_reconstruction_pilot_v1',started_unix=time.time(),
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in
                                 (REFERENCE,PREREG,ROOT/'model.py',ROOT/'deep_model.py',CHECKPOINT)},
                  environment=dict(torch=torch.__version__,cuda_build=torch.version.cuda),
                  limits=dict(tensor_bytes=256*2**20,feature_columns=256,watchdog_seconds=1800),phases={})
    try:
        with torch.inference_mode():
            t=time.perf_counter()
            result['phases']['A']=phase_a(R,torch)
            result['phases']['A']['seconds']=time.perf_counter()-t
            log(f"phase A: {result['phases']['A']['passed']}; max logit error {result['phases']['A']['max_logit_abs']}")
            if not result['phases']['A']['passed']:
                raise ArithmeticError('phase A closure failed; no discovery permitted')
            t=time.perf_counter()
            result['phases']['B']=phase_b(R,torch)
            result['phases']['B']['seconds']=time.perf_counter()-t
            if not result['phases']['B']['passed']:
                raise ArithmeticError('phase B controls failed; no trained-model promotion')
            t=time.perf_counter()
            result['phases']['C']=phase_c(R,torch)
            result['phases']['C']['seconds']=time.perf_counter()-t
            if torch.cuda.is_available():
                result['environment'].update(gpu=torch.cuda.get_device_name(),
                                             peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        result['predictions']={
            'pred_a_faithful_execution':result['phases']['A']['passed'],
            'pred_b_sound_discovery_controls':result['phases']['B']['passed'],
            'pred_c_contextual_execution':result['phases']['C']['passed'],
            'pred_d_reusable_computation_discovered':False,
        }
        result['verdict']='stop_local_discovery_method_retain_faithful_compiler'
        if result['phases']['C']['status']=='local_sharing_candidate_requires_followup':
            result['verdict']='candidate_only_not_a_discovered_circuit'
        if not result['phases']['C']['passed']:
            result['verdict']='contextual_execution_not_validated'
    except Exception as execution_exception:
        result['execution_error']=f'{type(execution_exception).__name__}: {execution_exception}'
        result['verdict']='invalid_or_incomplete_instrument'
        raise
    finally:
        signal.alarm(0)
        result['wall_seconds']=time.perf_counter()-started
        result['process_cpu_seconds']=time.process_time()-time_cpu
        result['bilin18_attention_cost_T512']=R.attention_cost(1,18,9,512,128)
        with OUT.open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        log(json.dumps({k:result[k] for k in ('verdict','wall_seconds','process_cpu_seconds')}))


if __name__=='__main__':
    main()
