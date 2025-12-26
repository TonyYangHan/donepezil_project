#!/bin/bash
set -euo pipefail

dir_name="processed_regions/zoom6"
script_name="lipid_analysis_accl_combined.py"

# Analyze all conditions together; the Python script will run all pairwise comparisons internally.
dirs=(
	"../${dir_name}/day7/0.1/"
	"../${dir_name}/day7/1mm/"
	"../${dir_name}/day7/cl/"
	"../${dir_name}/day35/0.1/"
	"../${dir_name}/day35/1mm/"
	"../${dir_name}/day35/cl/"
)

conds=(
	"day7_0.1"
	"day7_1mm"
	"day7_cl"
	"day35_0.1"
	"day35_1mm"
	"day35_cl"
)

out_dir="../${dir_name}/all_conditions/"

python "$script_name" "${dirs[@]}" -c "${conds[@]}" -o "$out_dir" --hide-ns -p "${out_dir}/all_conditions_plots.pdf"

echo "Done processing all conditions -> $out_dir"
