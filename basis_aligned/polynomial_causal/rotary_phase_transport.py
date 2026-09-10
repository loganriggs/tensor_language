"""Content-preserving transport through the actual rounded rotary matrices.

Last axis is native half-split (first half, second half). Tables must broadcast
to that half. This is a local phase intervention, not a text position edit.
"""
import torch


def rotate(x, cosine, sine):
    c, s = cosine.to(x), sine.to(x)
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a*c+b*s, -a*s+b*c), dim=-1)


def unrotate(x, cosine, sine):
    c, s = cosine.to(x), sine.to(x)
    determinant = c*c+s*s
    if not bool(torch.isfinite(determinant).all()) or bool((determinant <= 0).any()):
        raise ValueError('Noninvertible or nonfinite rotary pair')
    a, b = x.chunk(2, dim=-1)
    return torch.cat(((a*c-b*s)/determinant, (a*s+b*c)/determinant), dim=-1)


def transport(x, old_cosine, old_sine, new_cosine, new_sine):
    return rotate(unrotate(x, old_cosine, old_sine), new_cosine, new_sine)


def pair_dot(q, k, query_cosine, query_sine, key_cosine, key_sine):
    """Fold geometry into two scalar coefficients per frequency; no rotations."""
    cq, sq, ck, sk = (v.to(q) for v in (query_cosine, query_sine, key_cosine, key_sine))
    qa, qb = q.chunk(2, dim=-1)
    ka, kb = k.chunk(2, dim=-1)
    cosine = cq*ck+sq*sk
    sine = cq*sk-sq*ck
    return ((qa*ka+qb*kb)*cosine+(qa*kb-qb*ka)*sine).sum(-1)


def controls():
    from jacclust.tt_model import Rotary, apply_rotary_emb
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910437)
        # Use actual native table construction, width128, all512 positions.
        raw = torch.randn(2,512,3,128,dtype=torch.float64)
        k = torch.randn_like(raw)
        q2 = torch.randn_like(raw)
        rotary = Rotary(128)
        c,s = rotary(raw)
        native = apply_rotary_emb(raw,c,s)
        new = torch.arange(512).roll(7)
        nc,ns = c[:,new],s[:,new]
        oracle = apply_rotary_emb(raw,nc,ns)
        err = lambda a,b: float((a-b).abs().max())
        errors = {'rotate_oracle':err(rotate(raw,c,s),native),
                  'inverse':err(unrotate(native,c,s),raw),
                  'transport':err(transport(native,c,s,nc,ns),oracle),
                  'round_trip':err(transport(oracle,nc,ns,c,s),native)}
        j = torch.arange(512).roll(31)
        kc,ks = c[:,j],s[:,j]
        expected = (native*apply_rotary_emb(k,kc,ks)).sum(-1)
        folded = pair_dot(raw,k,c,s,kc,ks)
        errors['pair_dot'] = err(folded,expected)
        direct_product = expected*(apply_rotary_emb(q2,c,s)*apply_rotary_emb(k,nc,ns)).sum(-1)/128**2
        folded_product = folded*pair_dot(q2,k,c,s,nc,ns)/128**2
        errors['two_score_product'] = err(direct_product,folded_product)
        wrong = rotate(native,c,-s)  # transpose, incorrectly treated as inverse
        wrong_inverse_error = err(wrong,raw)
        det = c.double().square()+s.double().square()
        # Same displacement at different absolute positions need not match in BF16.
        exact_pc = c[:,5:].double()*c[:,:-5].double()+s[:,5:].double()*s[:,:-5].double()
        ps = c[:,5:].double()*s[:,:-5].double()-s[:,5:].double()*c[:,:-5].double()
        stationarity_error = max(float((exact_pc-exact_pc[:,0:1]).abs().max()),
                                 float((ps-ps[:,0:1]).abs().max()))
        f=raw.float();n=apply_rotary_emb(f,c,s)
        fp32_error=err(transport(n,c,s,nc,ns),apply_rotary_emb(f,nc,ns))
        assert max(errors.values()) < 1e-9 and fp32_error < 1e-5
        assert wrong_inverse_error > 1e-4 and stationarity_error > 1e-4
        return {'passed':True,'fp64_errors':errors,'fp32_transport_max_abs':fp32_error,
                'transpose_as_inverse_counterexample':wrong_inverse_error,
                'native_rotation_squared_scale_range':[float(det.min()),float(det.max())],
                'same_gap_absolute_phase_coefficient_difference':stationarity_error,
                'native_table_dtype':str(c.dtype),'positions':512,'head_width':128,
                'trained_model_forwards':0,'gpu_accessed':False,
                'scope':'Native rotary primitive controls; no learned distance mechanism or circuit identified.'}


if __name__ == '__main__':
    import json
    print(json.dumps(controls(),indent=2))
