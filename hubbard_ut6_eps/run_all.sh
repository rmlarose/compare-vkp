#!/bin/bash

subspace_file=hubbard_subspace_3.hdf5
if [[ ! -e $subspace_file ]]; then
    echo "${subspace_file} not found!"
    return 1
fi

input_files=($(ls new_input_*.json))

for i in $(seq 0 $((${#input_files[@]} - 1))); do
    python ../hubbard_eigenvalues.py ${input_files[i]} hubbard_subspace_3.hdf5 output_${i}.hdf5
done