#!/bin/bash
set -euo pipefail

parent_dir="../../processed_hyper_0331/visualization/lipid_24_5_tiff/"
script="prm_abundance.py"
out_dir="${parent_dir}abundance_results/"

inputs=(
    "${parent_dir}day7_cl/" "${parent_dir}day7_1mm/"
    "${parent_dir}day35_cl/" "${parent_dir}day35_1mm/"
)

conds=(
    "day7_cl" "day7_1mm"
    "day35_cl" "day35_1mm"
)

if [[ ${#inputs[@]} -ne ${#conds[@]} ]]; then
    echo "Error: inputs and conds must have the same length." >&2
    exit 1
fi

for dir in "${inputs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "Error: missing directory $dir" >&2
        exit 1
    fi
done

mkdir -p "$out_dir"

python "$script" \
    "${inputs[@]}" \
    --conds "${conds[@]}" \
    --out "$out_dir" \
    --pdf-out

