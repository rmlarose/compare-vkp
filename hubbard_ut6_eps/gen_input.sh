#!/bin/bash

eps=(1e-12 1e-10 1e-8 1e-7 1e-6 1e-4 1e-3 1e-2)

for i in $(seq 0 $((${#eps[@]} - 1))); do
    jq ".eps = ${eps[$i]}" input_3.json > new_input_${i}.json
done