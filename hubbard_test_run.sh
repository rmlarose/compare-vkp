#!/bin/bash

set -e

echo "*** DOING ED ***"
python hubbard_exact.py hubbard_input.json hubbard_ed_exact.hdf5
echo "*** DOING DMRG ***"
python hubbard_dmrg.py hubbard_input.json hubbard_exact.hdf5
echo "*** DOING RTE KRYLOV ***"
mpirun -n 12 python hubbard_subspace_matrices.py hubbard_input.json hubbard_exact.hdf5 hubbard_subspace.hdf5
python hubbard_eigenvalues.py hubbard_input.json hubbard_subspace.hdf5 hubbard_eigenvalues.hdf5
