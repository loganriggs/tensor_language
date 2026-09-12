"""Exact source-degree coefficient norms for shared quadratic products."""
import torch

def graded_norms(forms, producer_kernel, writer_gram):
    a=forms;b=forms@producer_kernel
    tr=torch.stack([torch.einsum('iab,jba->ij',a,a),
                    torch.einsum('iab,jba->ij',a,b)+torch.einsum('iab,jba->ij',b,a),
                    torch.einsum('iab,jba->ij',b,b)])
    parent=[a[0]@a[0],a[0]@b[0]+b[0]@a[0],b[0]@b[0]]
    total=forms.new_zeros(5)
    for i in range(1,len(forms)):
        for j in range(1,len(forms)):
            partner=[a[i]@a[j],a[i]@b[j]+b[i]@a[j],b[i]@b[j]]
            values=[forms.new_zeros(()) for _ in range(5)]
            for u in range(3):
                for v in range(3):
                    cycle=(parent[u]*partner[v].T).sum()
                    values[u+v]=values[u+v]+(tr[u,0,0]*tr[v,i,j]+tr[u,0,i]*tr[v,0,j]+4*cycle)/6
            total=total+torch.stack(values)*writer_gram[i-1,j-1]
    return total

def retained_grades(forms, root, directions, writer_gram):
    z=root@directions
    return graded_norms(forms,z@z.T,writer_gram)

def balanced_loss(forms, root, directions, writer_gram, full_grades):
    assert bool((full_grades[1:]>0).all())
    return (1-retained_grades(forms,root,directions,writer_gram)[1:]/full_grades[1:]).mean()
