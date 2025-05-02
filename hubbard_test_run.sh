#!/bin/bash

tau=1e-2
steps=1000
d=25
eps=1e-12

python hubbard_exact.py
python hubbard_subspace_matrices.py $tau $steps $d hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5 $eps
python hubbard_bound.py hubbard_exact.h5 hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5 hubbard_bound.hdf5 
