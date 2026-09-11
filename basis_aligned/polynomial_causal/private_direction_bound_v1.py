"""Numerical global bound for one discarded direction of a quadratic core.

Minimize 2 v.T A v - (v.T B v)^2, ||v||=1, by a scalar eigenvalue
envelope. Concavity chords bound each interval from below. Bounds use FP64
eigensolves with a stated roundoff margin, not formal interval arithmetic.
"""
import heapq
import math
import time

import torch


def optimize(core, shared_count, allowed_gap, max_splits=5000, seconds=30):
    scale = core.norm()
    normalized = core/scale
    a = (normalized@normalized)[shared_count:, shared_count:]
    b = normalized[shared_count:, shared_count:]
    ends = torch.linalg.eigvalsh(b)
    lo, hi = float(ends[0]), float(ends[-1])
    tolerance = allowed_gap/float(scale.square())
    guard = 1e-12
    best, vector, evaluations = math.inf, None, 0
    def evaluate(t):
        nonlocal best, vector, evaluations
        values, vectors = torch.linalg.eigh(2*a-2*t*b)
        v = vectors[:, 0]
        value = float(2*(v@a@v)-(v@b@v).square())
        if value<best:
            best, vector = value, v.clone()
        evaluations += 1
        return float(values[0])
    left, right = evaluate(lo), evaluate(hi)
    heap = []
    serial = 0
    def push(x, y, fx, fy):
        nonlocal serial
        if y-x<=1e-15:
            lower = min(x*x+fx, y*y+fy)-guard
        else:
            slope = (fy-fx)/(y-x)
            point = min(y, max(x, -slope/2))
            lower = point*point+fx+slope*(point-x)-guard
        serial += 1
        heapq.heappush(heap, (lower, serial, x, y, fx, fy))
    push(lo, hi, left, right)
    start, splits = time.perf_counter(), 0
    while heap and best-heap[0][0]>tolerance and splits<max_splits and time.perf_counter()-start<seconds:
        lower, _, x, y, fx, fy = heapq.heappop(heap)
        if lower>=best:
            continue
        mid = (x+y)/2
        fm = evaluate(mid)
        push(x, mid, fx, fm)
        push(mid, y, fm, fy)
        splits += 1
    lower = min(best, heap[0][0]) if heap else best
    gap = max(0., best-lower)*float(scale.square())
    keep = torch.linalg.qr(vector[:, None], mode='complete').Q[:, 1:]
    embedding = core.new_zeros(len(core), shared_count+keep.shape[1])
    embedding[:shared_count, :shared_count] = torch.eye(shared_count)
    embedding[shared_count:, shared_count:] = keep
    projected_loss = float(core.square().sum()-(embedding.T@core@embedding).square().sum())
    exact_loss = best*float(scale.square())
    replay = abs(projected_loss-exact_loss)/float(scale.square())
    return keep, dict(converged=gap<=allowed_gap, absolute_loss_gap=gap,
                      absolute_loss_upper=exact_loss,
                      absolute_loss_lower=lower*float(scale.square()),
                      normalized_projection_replay=replay, splits=splits,
                      eigenvalue_evaluations=evaluations, seconds=time.perf_counter()-start,
                      normalized_roundoff_margin=guard,
                      bound_scope='FP64 eigenvalue/chord numerical bound with 1e-12 scaled margin; not a formal interval-arithmetic certificate.')
