#!/bin/bash
set -euo pipefail

# This mirrors the directory/condition layout used in run_lipid_combined.sh
# and runs zoom6_corr.py to plot cross-condition correlations.

dir_name="processed_regions/zoom6_2"
script_name="zoom6_corr.py"

# Input directories per condition (must align with conds array below)
dirs=(
    "../${dir_name}/day7/0.1/"
    "../${dir_name}/day7/1mm/"
    "../${dir_name}/day7/cl/"
    "../${dir_name}/day35/0.1/"
    "../${dir_name}/day35/1mm/"
    "../${dir_name}/day35/cl/"
)

conds=(
    "day7_0.1"
    "day7_1mm"
    "day7_cl"
    "day35_0.1"
    "day35_1mm"
    "day35_cl"
)

# out_dir="../${dir_name}/all_conditions_corr/"
# pdf_out="${out_dir}/all_condition_correlations.pdf"
out_dir="../${dir_name}/all_conditions_corr_mean/"
pdf_out="${out_dir}/all_condition_correlations_mean.pdf"

mkdir -p "$out_dir"

# python "$script_name" "${dirs[@]}" --conds "${conds[@]}" --out "$out_dir" \
#     --pdf-out "$pdf_out" -pr --stat mean --min-value 0.01
python "$script_name" "${dirs[@]}" --conds "${conds[@]}" --out "$out_dir" \
    --pdf-out "$pdf_out"

echo "Done plotting correlations -> $out_dir"
