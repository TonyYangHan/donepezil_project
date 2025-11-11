#!/bin/bash
set -e

# Run lipid analysis over processed_regions/zoom6 using the same subfolder layout and output names.
dir_name="processed_regions/zoom6"
script_name="lipid_analysis_accl.py"

# Cross-day comparisons
python $script_name ../${dir_name}/day7/0.1/ \
    ../${dir_name}/day35/0.1/ day7_0.1 day35_0.1 -o ../${dir_name}/day7_vs_day35_0.1/
echo "Done processing day7 vs day35 0.1"

python $script_name ../${dir_name}/day7/1mm/ \
    ../${dir_name}/day35/1mm/ day7_1mm day35_1mm -o ../${dir_name}/day7_vs_day35_1mm/
echo "Done processing day7 vs day35 1mm"

python $script_name ../${dir_name}/day7/cl/ \
    ../${dir_name}/day35/cl/ day7_cl day35_cl -o ../${dir_name}/day7_vs_day35_cl/
echo "Done processing day7 vs day35 cl"

# Within-day comparisons: unique condition pairs per day
conds=("0.1" "1mm" "cl")
for day in day7 day35; do
    for ((i=0; i<${#conds[@]}-1; i++)); do
        for ((j=i+1; j<${#conds[@]}; j++)); do
            c1=${conds[$i]}
            c2=${conds[$j]}
            in1=../${dir_name}/${day}/${c1}/
            in2=../${dir_name}/${day}/${c2}/
            out=../${dir_name}/${c1}_vs_${c2}_${day}/
            echo "Processing ${c1} vs ${c2} ${day}"
            python $script_name "$in1" "$in2" ${day}_${c1} ${day}_${c2} -o "$out"
            echo "Done processing ${c1} vs ${c2} ${day}"
        done
    done
done