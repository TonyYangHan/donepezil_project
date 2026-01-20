#!/bin/bash
set -e

dir_name="processed_regions/zoom6_2"
script_name="calculate_ratio_LD_only.py"

# # All conditions
# python $script_name \
# 	../${dir_name}/day7/0.1/ \
# 	../${dir_name}/day7/1mm/ \
# 	../${dir_name}/day7/cl/ \
# 	../${dir_name}/day35/0.1/ \
# 	../${dir_name}/day35/1mm/ \
# 	../${dir_name}/day35/cl/ \
# 	-c day7_0.1 day7_1mm day7_cl day35_0.1 day35_1mm day35_cl \
# 	-o ../${dir_name}/all_conditions/turnover_medians.pdf

# day7 only (0.1 excluded)
python $script_name \
	../${dir_name}/day7/1mm/ \
	../${dir_name}/day7/cl/ \
	-c day7_1mm day7_cl \
	-o ../${dir_name}/plots_day7/turnover_medians.pdf

# day35 only (0.1 excluded)
python $script_name \
	../${dir_name}/day35/1mm/ \
	../${dir_name}/day35/cl/ \
	-c day35_1mm day35_cl \
	-o ../${dir_name}/plots_day35/turnover_medians.pdf

echo "Done processing LD-only turnover medians"
