#!/bin/bash

data_dir="krylov_ratio_copy"
for df in $(ls $data_dir/hubbard_subspace_*.hdf5); do
    # Get names of eigenvalue and merged files.
    eigv_file=${df/subspace/eigenvalues}
    if [[ ! -e $eigv_file ]]; then
        echo "${eigv_file} does not exist."
        exit 1
    fi
    merged_file=${df/subspace/merged}
    python merge_files.py $df $eigv_file $merged_file
done