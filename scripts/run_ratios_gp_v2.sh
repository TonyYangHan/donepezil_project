#!/bin/bash
set -e

dir_name="processed_regions/zoom6_2"
script_name="calculate_ratios_gp_v2.py"

# day7 only (0.1 excluded)
python $script_name \
  ../${dir_name}/day7/cl/ \
  ../${dir_name}/day7/1mm/ \
  -c day7_cl day7_1mm \
  -d -s -r -u -o ../${dir_name}/plots_day7/ \
  --hide-ns \
  -m -mw ../unet_segmentation/best_student_supervised.pth \
  -v -p -t 2 -i

# day35 only
python $script_name \
  ../${dir_name}/day35/cl/ \
  ../${dir_name}/day35/1mm/ \
  -c day35_cl day35_1mm \
  -d -s -r -u -o ../${dir_name}/plots_day35/ \
  --hide-ns \
  -m -mw ../unet_segmentation/best_student_supervised.pth \
  -v -p -t 2 -i

echo "Done processing all conditions"