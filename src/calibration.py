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
    cal_acc, cal_brier, cal_cls_ece, cal_ece, cal_kde, cal_nll
from utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def eval_method(logits, labels, idx):

    eval_result = {}
    
    acc = cal_acc(logits, labels, idx)
    nll = cal_nll(logits, labels, idx)
    brier = cal_brier(logits, labels, idx)
    ece = cal_ece(logits, labels, idx)
    kde = cal_kde(logits, labels, idx)
    cls_ece = cal_cls_ece(logits, labels, idx)

    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})

    return eval_result


def main(split, init, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = defaultdict(list)
    cal_val_result = defaultdict(list)
    cal_test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        # Load data
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)
        labels = data.y

        # Load model
        model = create_model(dataset, args).to(device)
        model_name = name_model(fold, args)
        dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
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


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type = 'Nodewise'
    max_splits,  max_init = 5, 5

    uncal_test_total = defaultdict(list)
    cal_val_total = defaultdict(list)
    cal_test_total = defaultdict(list)
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            (uncal_test_result,
             cal_val_result,
             cal_test_result) = main(split, init, args)
            for eval_metric in uncal_test_result:
                uncal_test_total[eval_metric].extend(uncal_test_result[eval_metric])
                cal_val_total[eval_metric].extend(cal_val_result[eval_metric])
                cal_test_total[eval_metric].extend(cal_test_result[eval_metric])

    val_mean = metric_mean(cal_val_total)
    # validate calibrator
    print(f"Val NNL: &{val_mean['nll']:.4f}")

    # print results
    for name, result in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total]):
        # print(name)
        test_mean = metric_mean(result)
        test_std = metric_std(result)
        print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                            f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                            f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                            f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                            f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                            f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")