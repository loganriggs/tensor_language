"""Exact 3x3 dyadic obstruction to common squared channels with positive Gram."""
from fractions import Fraction


def rational(matrix):
    return [[x if isinstance(x,Fraction) else Fraction(x) for x in row] for row in matrix]


def transpose(a):return [list(row) for row in zip(*a)]
def multiply(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def subtract(a,b):return [[x-y for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def determinant(a):
    if len(a)==1:return a[0][0]
    return sum((-1)**j*a[0][j]*determinant([row[:j]+row[j+1:] for row in a[1:]]) for j in range(len(a)))


def adjugate(a):
    n=len(a)
    cofactors=[[(-1)**(i+j)*determinant([row[:j]+row[j+1:] for k,row in enumerate(a) if k!=i]) for j in range(n)] for i in range(n)]
    return transpose(cofactors)


def obstruction(gram,forms):
    g=rational(gram);matrices=[rational(a) for a in forms]
    assert len(g)==3 and all(len(row)==3 for row in g)
    assert all(a==transpose(a) for a in [g]+matrices)
    minors=[determinant([row[:n] for row in g[:n]]) for n in (1,2,3)]
    if not all(v>0 for v in minors):raise ValueError('stored Gram is not positive definite')
    adj=adjugate(g);count=0
    for i,a in enumerate(matrices):
        for j in range(i+1,len(matrices)):
            b=matrices[j];count+=1
            comm=subtract(multiply(multiply(a,adj),b),multiply(multiply(b,adj),a))
            for row in range(3):
                for col in range(3):
                    if comm[row][col]:
                        return {'simultaneously_diagonalizable':False,'pairs_checked':count,
                            'positive_principal_minors':[str(v) for v in minors],
                            'witness':{'forms':[i,j],'entry':[row,col],'value':str(comm[row][col])}}
    return {'simultaneously_diagonalizable':True,'pairs_checked':count,
            'positive_principal_minors':[str(v) for v in minors],'witness':None}


def controls():
    identity=[[1,0,0],[0,1,0],[0,0,1]];diagonal=[[1,0,0],[0,2,0],[0,0,3]]
    other=[[2,0,0],[0,5,0],[0,0,-1]];w=[[1,2,0],[0,1,1],[0,0,1]]
    transform=lambda a:multiply(multiply(transpose(w),a),w)
    positive=obstruction(transform(identity),[transform(diagonal),transform(other)])
    negative=obstruction(identity,[diagonal,[[1,1,0],[1,0,0],[0,0,1]]])
    checks={'nonorthogonal_common_basis':positive['simultaneously_diagonalizable'],
            'noncommuting_negative':not negative['simultaneously_diagonalizable']}
    for label,g in (('singular',[[1,0,0],[0,1,0],[0,0,0]]),('indefinite',[[1,0,0],[0,-1,0],[0,0,1]])):
        try:obstruction(g,[diagonal]);checks[label+'_rejected']=False
        except ValueError:checks[label+'_rejected']=True
    return {'passed':all(checks.values()),'checks':checks,'negative_witness':negative['witness']}
