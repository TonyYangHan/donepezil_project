#!/bin/bash
set -e

python calculate_ratios_gp.py ../processed_data_0916/day7/0.1/ \
    ../processed_data_0916/day35/0.1/ -d --cond1 day7_0.1 --cond2 day35_0.1 -m -s -r -t -p -q -o ../processed_data_0916/day7_vs_day35_0.1/
echo "Done processing day7 vs day35 0.1"

python calculate_ratios_gp.py ../processed_data_0916/day7/1mm/ \
    ../processed_data_0916/day35/1mm/ -d --cond1 day7_1mm --cond2 day35_1mm -m -s -r -t -p -q -o ../processed_data_0916/day7_vs_day35_1mm/
echo "Done processing day7 vs day35 1mm"

python calculate_ratios_gp.py ../processed_data_0916/day7/cl/ \
    ../processed_data_0916/day35/cl/ -d --cond1 day7_cl --cond2 day35_cl -m -s -r -t -p -q -o ../processed_data_0916/day7_vs_day35_cl/
echo "Done processing day7 vs day35 cl"

# Within-day comparisons: unique condition pairs per day
conds=("0.1" "1mm" "cl")
for day in day7 day35; do
    for ((i=0; i<${#conds[@]}-1; i++)); do
        for ((j=i+1; j<${#conds[@]}; j++)); do
            c1=${conds[$i]}
            c2=${conds[$j]}
            in1=../processed_data_0916/${day}/${c1}/
            in2=../processed_data_0916/${day}/${c2}/
            out=../processed_data_0916/${c1}_vs_${c2}_${day}/
            echo "Processing ${c1} vs ${c2} ${day}"
            python calculate_ratios_gp.py "$in1" "$in2" -d \
                --cond1 ${day}_${c1} --cond2 ${day}_${c2} \
                -m -s -r -t -p -q -o "$out"
            echo "Done processing ${c1} vs ${c2} ${day}"
        done
    done
done