#!/bin/bash
set -e

dir_name="../../processed_hyper_05_28_26"
script_name="../hsi_unsup_kmeans_v5.py"

# python $script_name \
#     ${dir_name}/day7/0.1/ \
#     ${dir_name}/day7/1mm/ \
#     ${dir_name}/day7/cl/ \
#     ${dir_name}/day35/0.1/ \
#     ${dir_name}/day35/1mm/ \
#     ${dir_name}/day35/cl/ \
#     --conds day7_0.1 day7_1mm day7_cl day35_0.1 day35_1mm day35_cl \
#     -n 10 -d 4\
#     -o ${dir_name}/all_conditions/ \
#     --pdf

python $script_name \
    ${dir_name}/day7/0.1/ \
    ${dir_name}/day7/1mm/ \
    --conds day7_0.1 day7_1mm \
    -n 4 -d 1\
    -o ${dir_name}/day7_0.1_vs_1mm/ \
    --pdf