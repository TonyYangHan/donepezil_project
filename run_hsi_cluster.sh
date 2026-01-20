#!/bin/bash
set -e

dir_name="processed_hyper_1229"
script_name="hsi_unsup_kmeans_v3_gp_combined.py"
baseline_path="../../utils/water_HSI_76.csv"
prefix="../${dir_name}/"

python $script_name \
    $baseline_path \
    ${prefix}day7/cl/ \
    ${prefix}day7/0.1/ \
    ${prefix}day7/1mm/ \
    ${prefix}day35/cl/ \
    ${prefix}day35/0.1/ \
    ${prefix}day35/1mm/ \
    --conds day7_cl day7_0.1 day7_1mm day35_cl day35_0.1 day35_1mm \
    -n 6 -d 2 \
    -o ${prefix}all_conditions/ \
    --pdf

# python $script_name \
#     $baseline_path \
#     ${prefix}day7/1mm/ \
#     ${prefix}day7/cl/ \
#     ${prefix}day35/1mm/ \
#     ${prefix}day35/cl/ \
#     --conds day7_1mm day7_cl day35_1mm day35_cl \
#     -n 6 -d 2 \
#     -o ${prefix}all_conditions_no_0.1/ \
#     --pdf