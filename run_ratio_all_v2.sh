#!/bin/bash
set -e
block_size=32
dir_name="processed_0921"

python calculate_ratios_gp_v2.py ../${dir_name}/day7/0.1/ \
    ../${dir_name}/day35/0.1/ -d --cond1 day7_0.1 --cond2 day35_0.1 -s -p -b --block-size $block_size -o ../${dir_name}/day7_vs_day35_0.1/
echo "Done processing day7 vs day35 0.1"

python calculate_ratios_gp_v2.py ../${dir_name}/day7/1mm/ \
    ../${dir_name}/day35/1mm/ -d --cond1 day7_1mm --cond2 day35_1mm -s -p -b --block-size $block_size -o ../${dir_name}/day7_vs_day35_1mm/
echo "Done processing day7 vs day35 1mm"

python calculate_ratios_gp_v2.py ../${dir_name}/day7/cl/ \
    ../${dir_name}/day35/cl/ -d --cond1 day7_cl --cond2 day35_cl -s -p -b --block-size $block_size -o ../${dir_name}/day7_vs_day35_cl/
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
            python calculate_ratios_gp_v2.py "$in1" "$in2" -d \
                --cond1 ${day}_${c1} --cond2 ${day}_${c2} \
                -s -p -b --block-size $block_size -o "$out"
            echo "Done processing ${c1} vs ${c2} ${day}"
        done
    done
done