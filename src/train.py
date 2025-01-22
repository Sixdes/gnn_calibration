import os
import math
import random
import abc
import gc
import copy
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch 
import torch.nn.functional as F
from torch import Tensor, LongTensor
from model.model import create_model
from utils import set_global_seeds, arg_parse, name_model, metric_mean, metric_std
from utils import \
    cal_acc, cal_brier, cal_cls_ece, cal_ece, cal_kde, cal_nll
from data.data_utils import load_data
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
    # Evaluation
    val_result = defaultdict(list)
    test_result = defaultdict(list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        epochs = 2000
        lr = 0.01 #0.05
        model_name = name_model(fold, args)
        
        # Early stopping
        patience = 100
        vlss_mn = float('Inf')
        vacc_mx = 0.0
        state_dict_early_model = None
        curr_step = 0
        best_result = {}

        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)
        labels = data.y

        model = create_model(dataset, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)
        
        # print(model)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask]) 
            loss.backward()
            optimizer.step()

            # Evaluation on traing and val set
            accs = []
            nlls = []
            briers = []
            eces = []
            with torch.no_grad():
                model.eval()
                logits = model(data.x, data.edge_index)
                log_prob = F.log_softmax(logits, dim=1).detach()

                for mask in [data.train_mask, data.val_mask]:
                    eval_result = eval_method(logits, labels, mask)
                    acc, nll, brier, ece = eval_result['acc'], eval_result['nll'], eval_result['bs'], eval_result['ece']
                    accs.append(acc); nlls.append(nll); briers.append(brier); eces.append(ece)                    

                ### Early stopping
                val_acc = acc; val_loss = nll
                if val_acc >= vacc_mx or val_loss <= vlss_mn:
                    if val_acc >= vacc_mx and val_loss <= vlss_mn:
                        state_dict_early_model = copy.deepcopy(model.state_dict())
                        b_epoch = i
                        best_result.update({'logits':logits,
                                            'acc':accs[1],
                                            'nll':nlls[1],
                                            'bs':briers[1],
                                            'ece':eces[1]})
                    vacc_mx = np.max((val_acc, vacc_mx)) 
                    vlss_mn = np.min((val_loss, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break
                if args.verbose:
                    print(f'Epoch: : {i+1:03d}, Accuracy: {accs[0]:.4f}, NNL: {nlls[0]:.4f}, Brier: {briers[0]:.4f}, ECE:{eces[0]:.4f}')
                    print(' ' * 14 + f'Accuracy: {accs[1]:.4f}, NNL: {nlls[1]:.4f}, Brier: {briers[1]:.4f}, ECE:{eces[1]:.4f}')
        
        eval_result = eval_method(best_result['logits'], labels, data.test_mask)
        acc, nll, brier, ece = eval_result['acc'], eval_result['nll'], eval_result['bs'], eval_result['ece']
        test_result['acc'].append(acc); test_result['nll'].append(nll); test_result['bs'].append(brier)
        test_result['ece'].append(ece)

        del best_result['logits']
        for metric in best_result:
            val_result[metric].append(best_result[metric])


        # print("best epoch is:", b_epoch)
        dir = Path(os.path.join('model2', args.dataset, args.split_type, 'split'+str(split), 
                                'init'+ str(init)))
        dir.mkdir(parents=True, exist_ok=True)
        file_name = dir / (model_name + '.pt')
        torch.save(state_dict_early_model, file_name)
    return val_result, test_result


if __name__ == '__main__':
    args = arg_parse()
    set_global_seeds(args.seed)
    max_splits,  max_init = 5, 5


    val_total_result = {'acc':[], 'nll':[]}
    test_total_result = {'acc':[], 'nll':[]}
    for split in range(max_splits):
        for init in range(max_init):
            val_result, test_result = main(split, init, args)
            for metric in val_total_result:
                val_total_result[metric].extend(val_result[metric])
                test_total_result[metric].extend(test_result[metric])

    val_mean = metric_mean(val_total_result)
    test_mean = metric_mean(test_total_result)
    test_std = metric_std(test_total_result)
    print(f"Val  Accuracy: &{val_mean['acc']:.2f} \t" + " " * 8 +\
            f"NLL: &{val_mean['nll']:.4f}")
    print(f"Test Accuracy: &{test_mean['acc']:.2f}\pm{test_std['acc']:.2f} \t" + \
            f"NLL: &{test_mean['nll']:.4f}\pm{test_std['nll']:.4f}")

