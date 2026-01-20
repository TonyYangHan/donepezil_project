#!/bin/bash
set -euo pipefail

# Run prm_process.py for the same directories used in run_hsi_cluster.sh.
# Configure include/exclude keys here. Override via env vars if needed.
INCLUDE_KEY="PE"
EXCLUDE_KEY="18"

day7_pf="../processed_hyper_1229_prm/day7/"
day35_pf="../processed_hyper_1229_prm/day35/"
SCRIPT="prm_process.py"

inputs=(
	"${day7_pf}1mm/1_hyper_out"
	"${day7_pf}cl/1_hyper_out" "${day7_pf}cl/3_hyper_out"
	"${day35_pf}1mm/1_hyper_out"
	"${day35_pf}cl/1_hyper_out"
)

for dir in "${inputs[@]}"; do
	if [[ ! -d "$dir" ]]; then
		echo "Skipping missing directory: $dir" >&2
		continue
	fi

	echo "Normalizing TIFFs in $dir"
	python "$SCRIPT" \
		--input-dir "$dir" \
		--include-key "$INCLUDE_KEY" \
		--exclude-key "$EXCLUDE_KEY"
done
