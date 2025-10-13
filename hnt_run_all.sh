#!/bin/bash

set -e

python hubbard_exact.py hubbard_input.json hubbard_scratch/hubbard_exact.hdf5
python hubbard_non_toeplitz.py hubbard_input.json hubbard_scratch/hubbard_exact.hdf5 hubbard_scratch/subspace.hdf5
python hubbard_eigenvalues.py hubbard_scratch/eigval_input.json hubbard_scratch/subspace.hdf5 hubbard_scratch/eigvals.hdf5