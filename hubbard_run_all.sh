tau=0.05
d=16
eps=1e-12

for steps in 1 10 20 30 50 100 200 300 400
do
    subspace_file="data/hubbard_subspace_${steps}_steps.hdf5"
    eigvals_file="data/hubbard_eigvals_${steps}_steps.hdf5"
    python hubbard_subspace_matrices.py $tau $steps $d $subspace_file
    python hubbard_eigenvalues.py $subspace_file $eigvals_file $eps
done
