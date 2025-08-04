#!/bin/bash

set -e

mpirun --n 2 python phosphate_subspace.py phosphate_input.json . owp_631gd_22_ducc.data phosphate_output.hdf5