#!/bin/sh

# CoraFull
# Citeseer Pubmed  Computers Photo CS Physics 
for dataset in Cora 

do for calibration in TS 
# VS ETS CaGCN GATS

do for model in GCN
do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/calibration_edge_struct.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_delete \
	--config >> Log/modify_edge/calibration_all_modify_edge/${dataset}_${model}_delete.log

# PYTHONPATH=. python src/calibration_edge_struct.py --dataset $dataset \
#         --model $model \
#         --wdecay $wdecay \
#         --calibration $calibration \
#         --is_edge_add \
# 	--config >> Log/modify_edge/calibration_all_modify_edge/${dataset}_${model}_add.log
done
done
done