#!/bin/bash

set -e

data_dir="phosphate_data"
input_files=()
data_files=()
eigenvalue_files=()
for i in {0..2}; do
    input_files+=("${data_dir}/input_${i}.json")

    if [ ! -f ${input_files[$i]} ]; then
        echo "${input_files[$i]} does not exist."
    fi

    data_files+=("${data_dir}/phosphate_subspace_${i}.hdf5")

    if [ ! -f ${data_files[$i]} ]; then
        echo "${data_files[$i]} does not exist."
    fi

    eigenvalues_files+=("${data_dir}/phosphate_eigenvalues_${i}.json")
done

for  ((i = 0 ; i < ${#input_files[@]} ; i++)); do
    python hubbard_eigenvalues.py ${input_files[$i]} ${data_files[$i]} ${eigenvalues_files[$i]}
done
