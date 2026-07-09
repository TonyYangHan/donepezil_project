#!/bin/bash
set -e

dir_name="processed_1221"
script_name="calculate_ratio_masked_regions.py"

# day7 only
python $script_name \
	../${dir_name}/day7/cl/ \
	../${dir_name}/day7/1mm/ \
	-c day7_cl day7_1mm\
	-d -s --hide-ns \
	-o ../${dir_name}/plots_day7/ \
	-v -p
