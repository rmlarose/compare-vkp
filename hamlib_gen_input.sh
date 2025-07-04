#!/bin/bash

set -e

hamlib_files=(H2.hdf5 HF.hdf5 LiH.hdf5 O2.hdf5 OH.hdf5)
keys=("ham_JW-8" "ham_JW10" "ham_JW-10" "ham_JW16" "ham_JW12")
steps=(10 20 30 50 60 70 80 90 100 200)

if [[${#hamlib_files[@]} -ne ${#keys[@]}]]; then
    echo "Lengths of hamlib_files and keys do not match."
    exit 1
fi

for i in $(seq 0 $((${#hamlib_files[@]} - 1))); do
    jq '.hamlib_file |= "${hamlib_files[$i]}"' hamlib_input.json
done