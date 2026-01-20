#!/bin/bash
set -euo pipefail

# This mirrors the directory/condition layout used in run_lipid_combined.sh
# and runs zoom6_corr.py to plot cross-condition correlations.

dir_name="processed_regions/zoom6_2"
script_name="zoom6_corr.py"

# Input directories per condition (must align with conds array below)
# day7 (0.1 included)
dirs=(
    "../${dir_name}/day7/1mm/"
    "../${dir_name}/day7/cl/"
)

conds=(
    "day7_1mm"
    "day7_cl"
)

out_dir="../${dir_name}/plots_day7/"
pdf_out="${out_dir}/day7_correlations.pdf"

mkdir -p "$out_dir"

python "$script_name" "${dirs[@]}" --conds "${conds[@]}" --out "$out_dir" \
    --pdf-out "$pdf_out" -pr --min-value 0.01

# day35 (0.1 excluded)
dirs=(
    "../${dir_name}/day35/1mm/"
    "../${dir_name}/day35/cl/"
)

conds=(
    "day35_1mm"
    "day35_cl"
)

out_dir="../${dir_name}/plots_day35/"
pdf_out="${out_dir}/day35_correlations.pdf"

mkdir -p "$out_dir"

python "$script_name" "${dirs[@]}" --conds "${conds[@]}" --out "$out_dir" \
    --pdf-out "$pdf_out" -pr --min-value 0.01

echo "Done plotting correlations -> $out_dir"
