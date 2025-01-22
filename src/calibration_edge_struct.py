import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from collections import defaultdict
from data.data_utils import load_data, load_node_to_nearest_training
from model.model import create_model
from calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib
from utils import \
    cal_acc, cal_brier, cal_cls_ece, cal_ece, cal_kde, cal_nll, cal_conf
from utils import \
    set_global_seeds, arg_parse, name_model, \
    metric_mean, metric_std, default_cal_wdecay, plot_reliabilities

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def eval_method(logits, labels, idx):

    eval_result = {}
    
    acc = cal_acc(logits, labels, idx)
    nll = cal_nll(logits, labels, idx)
    brier = cal_brier(logits, labels, idx)
    ece = cal_ece(logits, labels, idx)
    conf = cal_conf(logits, labels, idx)
    kde = cal_kde(logits, labels, idx)
    cls_ece = cal_cls_ece(logits, labels, idx)

    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'conf':conf,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})

    return eval_result


def main(split, init, ratio, ratio_name, is_delete, is_add, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = defaultdict(list)
    cal_val_result = defaultdict(list)
    cal_test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        dir = Path(os.path.join('model_modify_edge_all', args.dataset, ratio_name, 'split'+str(split), 'init'+ str(init)))
        model_name = name_model(fold, args)
        file_name = dir / (model_name + '.pt')
        checkpoint = torch.load(file_name)

        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)
        labels = data.y

        # modify data
        if is_delete:
            new_edge_index = checkpoint['new_edge_index']
            print(f'the modify-delete {args.dataset} edges: {new_edge_index.size(1)} with ratio {ratio}') 
            data.edge_index = new_edge_index
            
        elif is_add:   
            new_edge_index =  checkpoint['new_edge_index']
            print(f'the modeify-add {args.dataset} edges: {new_edge_index.size(1)} with num {ratio}')
            data.edge_index = new_edge_index

        # Load model
        model = create_model(dataset, args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)

        eval_result = eval_method(logits, labels, data.test_mask)
        for metric in eval_result:
            uncal_test_result[metric].append(eval_result[metric])
        torch.cuda.empty_cache()

        ### Calibration
        if args.calibration == 'TS':
            temp_model = TS(model)
        elif args.calibration == 'IRM':
            temp_model = IRM(model)
        elif args.calibration == 'Spline':
            temp_model = SplineCalib(model, 7)
        elif args.calibration == 'Dirichlet':
            temp_model = Dirichlet(model, dataset.num_classes)
        elif args.calibration == 'OrderInvariant':
            temp_model = OrderInvariantCalib(model, dataset.num_classes)
        elif args.calibration == 'VS':
            temp_model = VS(model, dataset.num_classes)
        elif args.calibration == 'ETS':
            temp_model = ETS(model, dataset.num_classes)
        elif args.calibration == 'CaGCN':
            temp_model = CaGCN(model, data.num_nodes, dataset.num_classes, args.cal_dropout_rate)
        elif args.calibration == 'GATS':
            dist_to_train = load_node_to_nearest_training(args.dataset, args.split_type, split, fold)
            temp_model = GATS(model, data.edge_index, data.num_nodes, data.train_mask,
                            dataset.num_classes, dist_to_train, args.gats_args)
        
        
        ### Train the calibrator on validation set and validate it on the training set
        cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
        temp_model.fit(data, data.val_mask, data.train_mask, cal_wdecay)
        with torch.no_grad():
            temp_model.eval()
            logits = temp_model(data.x, data.edge_index)

        ### The training set is the validation set for the calibrator
        eval_result = eval_method(logits, labels, data.train_mask)
        for metric in eval_result:
            cal_val_result[metric].append(eval_result[metric])

        eval_result = eval_method(logits, labels, data.test_mask)
        for metric in eval_result:
            cal_test_result[metric].append(eval_result[metric])
        torch.cuda.empty_cache()
    return uncal_test_result, cal_val_result, cal_test_result

def per_ratio_ece(ratio, max_splits, max_init, is_delete=False, is_add=False, ratio_name=None, is_save_calfig=False):

    uncal_test_total = defaultdict(list)
    cal_val_total = defaultdict(list)
    cal_test_total = defaultdict(list)
    
    for split in range(max_splits):
        for init in range(max_init):
            (uncal_test_result,
             cal_val_result,
             cal_test_result) = main(split, init, ratio, ratio_name, is_delete, is_add, args)

            for eval_metric in uncal_test_result:
                uncal_test_total[eval_metric].extend(uncal_test_result[eval_metric])
                cal_val_total[eval_metric].extend(cal_val_result[eval_metric])
                cal_test_total[eval_metric].extend(cal_test_result[eval_metric])

    val_mean = metric_mean(cal_val_total)
    # validate calibrator
    print(f"Val NNL: &{val_mean['nll']:.4f}")
    test_mean_record = {'ece':[], 'acc':[], 'conf':[]}
    eval_type = 'Nodewise'

    # print results
    for name, result in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total]):
        # print(name)
        test_mean = metric_mean(result)
        test_std = metric_std(result)
        print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                            f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                            f"Conf: &{test_mean['conf']:.2f}$\pm${test_std['conf']:.2f} \t" + \
                            f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                            f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                            f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                            f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")
        # test_mean['ece'] with ratio
        test_mean_record['ece'].append(test_mean['ece'])
        test_mean_record['acc'].append(test_mean['acc'])
        test_mean_record['conf'].append(test_mean['conf'])
    
    return test_mean_record

def get_args(parser):
    # modify edge
    parser.add_argument('--is_edge_delete', action='store_true', default=False)
    parser.add_argument('--is_edge_add', action='store_true', default=False)

if __name__ == '__main__':
    args = arg_parse(get_args)
    print(args)
    set_global_seeds(args.seed)
    max_splits,  max_init = 3, 2
    delete_ratio_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    add_num_list = [0, 20, 50, 80, 110, 140, 170, 200, 230, 260, 300]
    
    delete_record_uncal = {'ece':[], 'acc':[], 'conf':[]}
    delete_record_cal = {'ece':[], 'acc':[], 'conf':[]}
    add_record_uncal = {'ece':[], 'acc':[], 'conf':[]}
    add_record_cal = {'ece':[], 'acc':[], 'conf':[]}

    print(f'-----------------------------------training with {args.model} on {args.dataset}-----------------------------------')
    if args.is_edge_delete:
        for delete_ratio in delete_ratio_list:
            ratio_name = 'delete_'+ str(delete_ratio)
            de_record = per_ratio_ece(delete_ratio, max_splits, max_init, is_delete=True, ratio_name=ratio_name)
            for metric in de_record.keys():
                delete_record_uncal[metric].append(de_record[metric][0])
                delete_record_cal[metric].append(de_record[metric][1])
            
        print('----------------------------------------------------------------------------')
        print(f'delete edge mean ece uncal : {delete_record_uncal["ece"]}')
        print(f'delete edge mean ece {args.calibration} : {delete_record_cal["ece"]}')
        print(f'delete edge mean acc uncal : {delete_record_uncal["acc"]}')
        print(f'delete edge mean acc {args.calibration} : {delete_record_cal["acc"]}')
        print(f'delete edge mean conf uncal : {delete_record_uncal["conf"]}')
        print(f'delete edge mean conf {args.calibration} : {delete_record_cal["conf"]}')
        print('----------------------------------------------------------------------------')

        # draw_del_ratio_ece(delete_ratio_list, delete_edge_ece_uncal, delete_edge_ece_cal, title='delete-Edge ECE Comparison', save_path='/root/GATS/figure/del_edge_ece.png')

    elif args.is_edge_add:
        for add_num in add_num_list:
            ratio_name = 'add_' + str(add_num)
            add_record = per_ratio_ece(add_num, max_splits, max_init, is_add=True, ratio_name=ratio_name)
            for metric in add_record.keys():
                add_record_uncal[metric].append(add_record[metric][0])
                add_record_cal[metric].append(add_record[metric][1])
        print('----------------------------------------------------------------------------')
        print(f'add edge mean ece uncal : {add_record_uncal["ece"]}')
        print(f'add edge mean ece {args.calibration} : {add_record_cal["ece"]}')
        print(f'add edge mean acc uncal : {add_record_uncal["acc"]}')
        print(f'add edge mean acc {args.calibration} : {add_record_cal["acc"]}')
        print(f'add edge mean conf uncal : {add_record_uncal["conf"]}')
        print(f'add edge mean conf {args.calibration} : {add_record_cal["conf"]}')
        # draw_del_ratio_ece(add_ratio_list, add_edge_ece_uncal, add_edge_ece_cal, title='Add-Edge ECE Comparison', save_path='/root/GATS/figure/add_edge_ece.png')
        print('----------------------------------------------------------------------------')

    