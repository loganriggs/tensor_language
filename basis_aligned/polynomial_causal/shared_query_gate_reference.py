"""Necessary exact linear-product condition and a coupled-but-shared control."""
from fractions import Fraction as F
import simultaneous_congruence_reference as S


def obstruction(forms):
    for index,form in enumerate(forms):
        a=S.rational(form);assert len(a)==3 and a==S.transpose(a)
        value=S.determinant(a)
        if value:
            return {'terminal':'no_common_linear_gate','form_index':index,'determinant':str(value)}
    return {'terminal':'inconclusive_all_forms_singular','form_index':None,'determinant':None}


def controls():
    from audit_query_coupled_block_bound_v1 import block_certificate
    g=[[1,0,0],[0,1,0],[0,0,1]]
    forms=[[[1,0,0],[0,0,0],[0,0,0]],
           [[0,F(1,2),0],[F(1,2),0,0],[0,0,0]],
           [[0,0,F(1,2)],[0,0,0],[F(1,2),0,0]]]
    evaluate=lambda z:[sum(z[i]*a[i][j]*z[j] for i in range(3) for j in range(3)) for a in forms]
    w=[[1,2,0],[0,1,1],[0,0,1]];transform=lambda a:S.multiply(S.multiply(S.transpose(w),a),w)
    checks={'shared_gate_not_rejected':obstruction(forms)['terminal']=='inconclusive_all_forms_singular',
            'shared_gate_despite_irreducible_block':block_certificate(g,forms)['scalar_only'],
            'nonorthogonal_shared_gate_not_rejected':obstruction([transform(a) for a in forms])['terminal']=='inconclusive_all_forms_singular',
            'full_rank_quadratic_rejected':obstruction([g])['terminal']=='no_common_linear_gate',
            'nonorthogonal_full_rank_rejected':obstruction([transform(g)])['terminal']=='no_common_linear_gate'}
    for index,z in enumerate(([F(1,3),F(-2,7),F(4,5)],[2,2,3],[0,2,3],[2,0,3],[2,2,0],[0,0,3])):
        checks['amplitude_and_joint_edits_'+str(index)]=evaluate(z)==[z[0]*x for x in z]
    return {'passed':all(checks.values()),'checks':checks,
            'scope':'Planted common multiplication only; trained route determinants unopened.'}
