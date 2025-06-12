#!/bin/bash

# Run the eigenvalue calculation with different values
# of epsilon, the threshold for the GEVP solver.
# It seems we can get very divergent energies if epsilon is too small.

set -e

eps=(1e-12 1e-10 1e-8 1e-7 1e-6 1e-4 1e-3 1e-2)

data_dir=hubbard_ut6_eps/hubbard_dmrg_4
subspace_files=($(ls ${data_dir}/hubbard_subspace_*.hdf5))
for i in $(seq 0 $((${#subspace_files[@]} - 1))); do
    input_file="input_${i}.json"
    if [[ ! -e $data_dir/$input_file ]]; then
        echo "File ${input_file} does not exist!"
        exit 1
    fi

    # Make new input files with different values of epsilon
    # and do the eigenvalue calculation for each.
    for j in $(seq 0 $((${#eps[@]} - 1))); do
        jq ".eps = ${eps[$j]}" ${data_dir}/input_${i}.json > ${data_dir}/new_input_${i}_${j}.json
        python hubbard_eigenvalues.py ${data_dir}/new_input_${i}_${j}.json ${subspace_files[$i]} ${data_dir}/new_eigenvalues_${i}_${j}.hdf5
    done
done