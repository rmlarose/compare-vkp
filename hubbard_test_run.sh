#!/bin/bash

tau=1e-1
steps=100
d=16
eps=1e-12

python hubbard_subspace_matrices.py $tau $steps $d hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5 $eps
