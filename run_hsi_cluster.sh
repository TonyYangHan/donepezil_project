#!/bin/bash

prefix="../processed_hyper/"
set -e
python hsi_unsup_kmeans_v3_gp_combined.py ../../utils/water_HSI_76.csv ${prefix}day7/ ${prefix}day35/ \
    -n 8 -d 2 -o ${prefix}combined/