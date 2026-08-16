"""Is the dictionary's higher attributable fraction about the BASIS, or about removing
less energy?

bilin18_joint_removal.py found, at 32 directions each:

    arm        removed energy   joint dCE   sum of solos   attributable
    svd32              70.5%      +0.3832        +0.0345           9.0%
    rot32              70.5%      +0.3832        +0.0350           9.1%
    dict32             64.3%      +0.1146        +0.0151          13.1%
    dict4096           53.0%      +0.1820        +0.0374          20.5%

read at face value as A5's prediction confirmed: dictionary atoms are 2.3x more
individually attributable than SVD directions. But the removed energies are NOT matched,
and the test specification required matching them precisely because they confound this
statistic -- a set of directions that removes less of the layer's output has less overlap
to interfere, so it should look more attributable for reasons that have nothing to do
with the basis being better. dict4096 removes 53% where svd32 removes 70.5%, and it is
the arm carrying the result.

So: hold energy fixed and vary the count instead. Two matched comparisons,

    svd at ~53% energy   (fewer than 32 directions)  vs  dict4096 at 32 (53%)
    dict4096 at ~70.5%   (more than 32 atoms)        vs  svd32 at 32   (70.5%)

If the gap survives both, the basis is doing the work and A5 is right. If svd rises to
~20% when it removes only 53%, and dict4096 falls to ~9% when it removes 70.5%, the
effect was energy all along and the original reading is wrong.
"""
import json, sys, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import (fwd, held, collect_out, fit_sae, orth, score,
                                   TRAIN, HELD, LAYER, NDIR, DEV, PATCH)

def main():
    base = held(); BASE=float(base.mean())
    Ytr = collect_out(TRAIN, LAYER); Ybar=Ytr.mean(0); Yc=Ytr-Ybar
    tot=float(Yc.pow(2).sum())
    _,Sv,Vh = torch.linalg.svd(Yc, full_matrices=False)
    a4k,u4k,l0,fvu,l1 = fit_sae(Ytr, 4096, target_l0=40)
    order = u4k.argsort(descending=True)
    prev = json.load(open('bilin18_joint_removal_results.json'))
    E_SVD = prev['arms']['svd32']['removed_energy']       # 0.705
    E_DICT = prev['arms']['dict4096']['removed_energy']   # 0.530

    def energy(Q): return float((Yc@Q).pow(2).sum())/tot
    def svd_at(r): return Vh[:r].T
    def dict_at(r): return a4k[order[:r]].T

    # find counts hitting the two target energies
    def find(builder, target, lo, hi):
        best=None
        for r in range(lo, hi+1, 2):
            Q=orth(builder(r)); e=energy(Q)
            if best is None or abs(e-target)<abs(best[1]-target): best=(r,e,Q)
            if e>target+0.02 and best[1]>=target: break
        return best

    r_svd, e_svd, Q_svd = find(svd_at, E_DICT, 4, 32)       # svd down at dict's energy
    r_dic, e_dic, Q_dic = find(dict_at, E_SVD, 34, 160)     # dict up at svd's energy
    out={'base_ce':BASE,'reference':prev['arms'],'matched':{}}
    print(f'dictionary refit: L0 {l0:.1f} FVU {fvu:.3f}\n')
    print(f"matching at {100*E_DICT:.1f}% energy: svd needs {r_svd} directions "
          f"(got {100*e_svd:.1f}%)")
    print(f"matching at {100*E_SVD:.1f}% energy: dict needs {r_dic} atoms "
          f"(got {100*e_dic:.1f}%)\n")
    print(f"  {'arm':>22} {'n':>5} {'energy':>8} {'joint':>9} {'solos':>9} {'attributable':>13}")
    for name,(r,e,Q) in (('svd @ dict energy',(r_svd,e_svd,Q_svd)),
                         ('dict @ svd energy',(r_dic,e_dic,Q_dic))):
        j,s,per = score(Q, Ybar, base)
        f=s/j if abs(j)>1e-9 else float('nan')
        out['matched'][name]={'n':r,'removed_energy':e,'joint_dce':j,'solo_sum':s,
                              'attributable_fraction':f}
        print(f"  {name:>22} {r:>5} {100*e:>7.1f}% {j:>+9.4f} {s:>+9.4f} {100*f:>12.1f}%",
              flush=True)
    # the two original points, for the table
    for k in ('svd32','dict4096'):
        a=prev['arms'][k]
        print(f"  {k+' (original)':>22} {32:>5} {100*a['removed_energy']:>7.1f}% "
              f"{a['joint_dce']:>+9.4f} {a['solo_sum']:>+9.4f} "
              f"{100*a['attributable_fraction']:>12.1f}%")
    lo=out['matched']['svd @ dict energy']['attributable_fraction']
    hi=out['matched']['dict @ svd energy']['attributable_fraction']
    sv=prev['arms']['svd32']['attributable_fraction']
    dc=prev['arms']['dict4096']['attributable_fraction']
    print(f"\nAT MATCHED ENERGY:")
    print(f"  ~{100*E_DICT:.0f}%: svd {100*lo:.1f}%  vs  dict {100*dc:.1f}%")
    print(f"  ~{100*E_SVD:.0f}%: svd {100*sv:.1f}%  vs  dict {100*hi:.1f}%")
    survives = (dc > 1.3*lo) and (hi > 1.3*sv)
    out['gap_survives_energy_matching']=bool(survives)
    out['verdict']=('the dictionary advantage is real and not an energy artefact'
                    if survives else
                    'the apparent dictionary advantage is largely an energy artefact')
    print(f"  -> {out['verdict']}")
    json.dump(out, open('bilin18_joint_removal_matched.json','w'), indent=1)
    print('\nwrote bilin18_joint_removal_matched.json')

if __name__=='__main__': main()
