#!/bin/bash
set -e

# Adjust dir_name if your processed folder differs
DIR_NAME="processed_1221"
SCRIPT="zoom1_corr.py"

python "$SCRIPT" \
    ../${DIR_NAME}/day7/0.1/ \
    ../${DIR_NAME}/day7/1mm/ \
    ../${DIR_NAME}/day7/cl/ \
    ../${DIR_NAME}/day35/0.1/ \
    ../${DIR_NAME}/day35/1mm/ \
    ../${DIR_NAME}/day35/cl/ \
    --conds day7_0.1 day7_1mm day7_cl day35_0.1 day35_1mm day35_cl \
    --out ../${DIR_NAME}/all_conditions_corr/ \
    --pdf-out ../${DIR_NAME}/all_conditions_corr/zoom1_corr_plots.pdf \
    --per-cond --min-value 0.01

echo "zoom1 correlation plots complete."