#!/bin/bash

set -e

subspace_files=($(ls data_copy | grep 'subspace'))

for sfile in "${subspace_files[@]}"; do
    ev_file=$(echo $sfile | sed -e 's/subspace/eigenvalues/')
    if [[ ! -e data_copy/$ev_file ]]; then
        echo "File $ev_file does not exist!"
        exit 1
    fi

    output_file=$(echo $sfile | sed -e 's/subspace/bound/')
    if [[ -e data_copy/$output_file ]]; then
        rm data_copy/$output_file
    fi
    log_file=$(echo $output_file | sed -e 's/hdf5/log/')
    if [[ -e data_copy/$log_file ]]; then
        rm data_copy/$log_file
    fi

    python hubbard_bound.py data_copy/hubbard_exact.h5 data_copy/$sfile data_copy/$ev_file data_copy/$output_file > data_copy/$log_file
done