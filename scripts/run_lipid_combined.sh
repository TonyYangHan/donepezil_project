#!/bin/bash
set -euo pipefail

dir_name="processed_regions/zoom6_2"
script_name="lipid_analysis_accl_combined.py"

# day7 only
dirs=(
	# "../${dir_name}/day7/0.1/"
	"../${dir_name}/day7/cl/"
	"../${dir_name}/day7/1mm/"
)

conds=(
	# "day7_0.1"
	"day7_cl"
	"day7_1mm"
)

out_dir="../${dir_name}/plots_day7/"

python "$script_name" "${dirs[@]}" -c "${conds[@]}" -o "$out_dir" \
	--hide-ns -a -p "${out_dir}/day7_lipid_plots.pdf" -x 7

# day35 only
dirs=(
	# "../${dir_name}/day35/0.1/"
	"../${dir_name}/day35/cl/"
	"../${dir_name}/day35/1mm/"
)

conds=(
	# "day35_0.1"
	"day35_cl"
	"day35_1mm"
)

out_dir="../${dir_name}/plots_day35/"

python "$script_name" "${dirs[@]}" -c "${conds[@]}" -o "$out_dir" \
	--hide-ns -a -p "${out_dir}/day35_lipid_plots.pdf" -x 2

echo "Done processing all conditions -> $out_dir"
