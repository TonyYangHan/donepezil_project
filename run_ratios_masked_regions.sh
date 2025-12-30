#!/bin/bash
set -e

dir_name="processed_1221"
script_name="calculate_ratio_masked_regions.py"

python $script_name \
	../${dir_name}/day7/0.1/ \
	../${dir_name}/day7/1mm/ \
	../${dir_name}/day7/cl/ \
	../${dir_name}/day35/0.1/ \
	../${dir_name}/day35/1mm/ \
	../${dir_name}/day35/cl/ \
	-c day7_0.1 day7_1mm day7_cl day35_0.1 day35_1mm day35_cl \
	-d -s --hide-ns \
	-o ../${dir_name}/all_conditions/ \
	-v -p

echo "Done processing masked regions"
