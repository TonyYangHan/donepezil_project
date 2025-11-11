#!/bin/bash
set -e

# Run lipid analysis over processed_regions/zoom6 using the same subfolder layout and output names.
dir_name="processed_0921"
script_name="redox_turnover_corr.py"

# Within-day comparisons: unique condition pairs per day
conds=("0.1" "1mm" "cl")
for day in day7 day35; do
    for cd in "${conds[@]}"; do
        echo "Processing ${cd} ${day}"
        in=../${dir_name}/${day}/${cd}/
        out=../${dir_name}/redox_turnover_new/
        python $script_name "$in" -b -p ${day}_${cd} -o "$out"
        echo "Done processing ${cd} ${day}"
    done
done