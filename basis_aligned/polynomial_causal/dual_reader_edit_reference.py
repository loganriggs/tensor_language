"""Minimum-norm independent writes for two fixed linear output readers."""
import numpy as np


def compile_readers(a,b):
    c=np.concatenate([a,b]);gram=c@c.T
    if np.linalg.matrix_rank(c)!=len(c):raise ValueError('dependent task readers')
    d=np.linalg.solve(gram,c).T
    return c,d


def norm_gain(c,d):
    own=c.T@np.linalg.inv(c@c.T);base=own.T@own;changed=d.T@d
    values,vectors=np.linalg.eigh(base);inverse=(vectors/np.sqrt(values))@vectors.T
    return float(np.sqrt(np.linalg.eigvalsh(inverse@changed@inverse)[-1]))


def diagnostics(a,b):
    c,d=compile_readers(a,b);n=len(a);da,db=d[:,:n],d[:,n:]
    pa,pb=da@a,db@b;eye=np.eye(len(c));joint=d@c
    q=np.linalg.qr(c.T,mode='reduced')[0]
    errors={'right_inverse':np.max(abs(c@d-eye)),
            'a_idempotence':np.max(abs(pa@pa-pa)), 'b_idempotence':np.max(abs(pb@pb-pb)),
            'a_after_b':np.max(abs(pa@pb)), 'b_after_a':np.max(abs(pb@pa)),
            'sum_joint_projector':np.max(abs(pa+pb-q@q.T)),
            'joint_idempotence':np.max(abs(joint@joint-joint))}
    qa=np.linalg.qr(a.T,mode='reduced')[0];qb=np.linalg.qr(b.T,mode='reduced')[0]
    return {'identity_max_abs':{k:float(v) for k,v in errors.items()},
            'task_principal_cosines':np.linalg.svd(qa.T@qb,compute_uv=False).tolist(),
            'reader_condition_number':float(np.linalg.cond(c)),
            'worst_write_norm_gain':{'temporal':norm_gain(a,da),'iswas':norm_gain(b,db)},
            'ordinary_patch_cross_reader_operator_norm':{'a_to_b':float(np.linalg.norm(b@(a.T@np.linalg.inv(a@a.T)),2)),
                                                        'b_to_a':float(np.linalg.norm(a@(b.T@np.linalg.inv(b@b.T)),2))}}


def controls():
    a=np.array([[1.,0.,0.]]);b=np.array([[.8,.6,0.]])
    c,d=compile_readers(a,b);r=diagnostics(a,b);x=np.array([2.,3.,4.]);amp=np.array([.3,-.7])
    sequential=x+d[:,0]*amp[0]+d[:,1]*amp[1];joint=x+d@amp
    pa=d[:,:1]@a;pb=d[:,1:]@b;i=np.eye(3)
    try:compile_readers(a,a);rejected=False
    except ValueError:rejected=True
    near=diagnostics(a,np.array([[np.sqrt(1-1e-6),1e-3,0.]]))
    checks={'independent_readers':np.max(abs(c@(joint-x)-amp))<1e-12,
            'joint_edits':np.max(abs(sequential-joint))<1e-12,
            'removals_commute':np.max(abs((i-pa)@(i-pb)-(i-pb)@(i-pa)))<1e-12,
            'ordinary_patch_cross_talk':r['ordinary_patch_cross_reader_operator_norm']['a_to_b']>.7,
            'rank_deficiency_rejected':rejected,'near_collinear_large_gain':near['worst_write_norm_gain']['temporal']>900}
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks}
