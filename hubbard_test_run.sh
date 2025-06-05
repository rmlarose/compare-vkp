#!/bin/bash

set -e

python hubbard_dmrg.py hubbard_input.json hubbard_exact.hdf5
python hubbard_subspace_matrices.py hubbard_input.json hubbard_exact.hdf5 hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_input.json hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5
