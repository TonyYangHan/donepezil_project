#!/bin/bash
set -e

dir_name="../../processed_05_28_26/zoom6"

python "../calculate_ratios_gp_v3.py" \
  ${dir_name}/day7/0.1/ ${dir_name}/day7/cl/ ${dir_name}/day7/1mm/ \
  -c day7_0.1 day7_cl day7_1mm\
  -d -m -s -pc 2.0 \
  -o ${dir_name}/plots_all_day7/

python "../calculate_ratios_gp_v3.py" \
  ${dir_name}/day35/0.1/ ${dir_name}/day35/cl/ ${dir_name}/day35/1mm/ \
  -c day35_0.1 day35_cl day35_1mm\
  -d -m -s -pc 2.0 \
  -o ${dir_name}/plots_all_day35/

python "../calculate_ratios_gp_v3.py" \
  ${dir_name}/day7/0.1/ ${dir_name}/day35/0.1/ \
  -c day7_0.1 day35_0.1 \
  -d -m -s -pc 2.0 \
  -o ${dir_name}/plots_0.1/

python "../calculate_ratios_gp_v3.py" \
  ${dir_name}/day7/1mm/ ${dir_name}/day35/1mm/ \
  -c day7_1mm day35_1mm \
  -d -m -s -pc 2.0 \
  -o ${dir_name}/plots_1mm/

python "../calculate_ratios_gp_v3.py" \
  ${dir_name}/day7/cl/ ${dir_name}/day35/cl/ \
  -c day7_cl day35_cl \
  -d -m -s -pc 2.0 \
  -o ${dir_name}/plots_cl/