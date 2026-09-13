"""Exact error identity when a shared inside projector is reduced."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
torch.manual_seed(7131130)
p1,p2,d1,d2,o1,o2=[torch.randn(7,9,dtype=torch.float64) for _ in range(6)]
original=(p1+d1)*(p2+d2)+o1*o2
approx=p1*p2+(o1+d1)*(o2+d2)
residual=(o1-p1)*d2+d1*(o2-p2)
err=float((approx-original-residual).norm()/(approx-original).norm())
# A dropped-dropped-only case is deliberately nonzero in the inside product,
# yet exactly cancels in the full even numerator.
only_dd_original=d1*d2;only_dd_approx=d1*d2
out={'pred_a':err<=1e-12,'pred_b':bool(torch.equal(only_dd_original,only_dd_approx)),'identity_relative_error':err,'inside_only_dropped_product_norm':float((d1*d2).norm()),'scope':'Exact scalar-pair identity before common denominators/value contraction. Does not claim dropped directions can be removed from required full key reads or normalizers.'}
(P/'EVEN_KEY_DISCARDED_PAIR_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
