#!/bin/bash
set -e

block_size=32
dir_name="processed_regions/zoom6_2"
script_name="calculate_ratios_gp_v2.py"

# All conditions in one run (multi-condition plotting/statistics)
python $script_name \
  ../${dir_name}/day7/0.1/ \
  ../${dir_name}/day7/1mm/ \
  ../${dir_name}/day7/cl/ \
  ../${dir_name}/day35/0.1/ \
  ../${dir_name}/day35/1mm/ \
  ../${dir_name}/day35/cl/ \
  -c day7_0.1 day7_1mm day7_cl day35_0.1 day35_1mm day35_cl \
  -d -s -r -u -o ../${dir_name}/all_conditions/ \
  --hide-ns \
  -m -mw ../unet_segmentation/best_student_supervised.pth \
  -v -p

echo "Done processing all conditions"