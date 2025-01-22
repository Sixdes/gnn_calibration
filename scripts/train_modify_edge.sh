#!/bin/sh

for dataset in Cora Citeseer 
# Pubmed Computers Photo CS Physics CoraFull

do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_delete \
        --wdecay $wdecay >> Log/modify_edge/train_all_modify_edge/${dataset}_delete.log

PYTHONPATH=. python src/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_add \
        --wdecay $wdecay >> Log/modify_edge/train_all_modify_edge/${dataset}_add.log
done
done



