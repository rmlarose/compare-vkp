#!/bin/bash

python hubbard_subspace_matrices.py 1e-1 300 20 hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5
