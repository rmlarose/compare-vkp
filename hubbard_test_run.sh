#!/bin/bash

set -e

n=2
tau=1e-1
steps=300
d=16
eps=1e-9
ratio=0.3

python hubbard_exact.py $n hubbard_exact.hdf5
python hubbard_subspace_matrices.py $n $tau $steps $d $ratio hubbard_exact.hdf5 hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5 $eps
