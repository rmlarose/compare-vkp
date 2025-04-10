tau=1e-1
d=16

for steps in 1 10 20 30 50 100
do
    subspace_file="data/hubbard_subspace_${steps}_steps.h5"
    eigvals_file="data/hubbard_eigvals_${steps}_steps.h5"
    python hubbard_subspace_matrices.py $tau $steps $d $subspace_file
    python hubbard_eigenvalues.py $subspace_file $eigvals_file
done