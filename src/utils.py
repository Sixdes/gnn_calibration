import os
import math
import random
import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Sequence
import torch.nn.functional as F
from KDEpy import FFTKDE
from typing import NamedTuple
from torch import nn, Tensor, LongTensor, BoolTensor


def cal_acc(logits, labels, idx):
    preds = torch.argmax(logits, dim=1)
    return torch.mean(
        (preds[idx] == labels[idx]).to(torch.get_default_dtype())
    ).item()

def ece_loss(confs, corrects, bins, norm):
    sortedconfs, sortindices = torch.sort(confs)
    binidx = (sortedconfs * bins).long()
    binidx[binidx == bins] = bins - 1
    bincounts = binidx.bincount(minlength=bins)
    bincumconfs = partial_sums(sortedconfs, bincounts)
    bincumcorrects = partial_sums(
        corrects[sortindices].to(dtype=torch.get_default_dtype()),
        bincounts)
    errs = (bincumconfs - bincumcorrects).abs() / (
            bincounts + torch.finfo().tiny)

    return ((errs ** norm) * bincounts / bincounts.sum()).sum()


def cal_ece(logits, labels, idx_test, bins=15, norm=1):
    logits, labels = logits[idx_test], labels[idx_test]
    confs, preds = torch.softmax(logits, -1).max(dim=-1)
    corrects = (preds == labels)
    ece_score = ece_loss(confs, corrects, bins, norm)

    return ece_score.item()


def cal_nll(logits, labels, idx_test):
    return F.cross_entropy(logits[idx_test], labels[idx_test]).item()

def cal_conf(logits, labels, idx_test):
    confs, _ = torch.softmax(logits[idx_test], -1).max(dim=-1)
    return confs.mean().item()


def cal_brier(logits, labels, idx_test):
    nodeprobs = torch.softmax(logits[idx_test], -1)
    nodeconfs = torch.gather(nodeprobs, -1, labels[idx_test].unsqueeze(-1)).squeeze(-1)
    return (nodeprobs.square().sum(dim=-1) - 2.0 * nodeconfs).mean().add(1.0).item()

def cal_cls_ece(logits, labels, idx_test, bins=15, norm=1):
    nodelogits, nodegts = logits[idx_test], labels[idx_test]
    nodeconfs = torch.softmax(nodelogits, -1)
    num_classes = logits.size(1)
    class_ece = torch.zeros(num_classes, device=logits.device)
    for i in range(num_classes):
        classconfs = nodeconfs[:,i]
        frequency = nodegts.eq(i)
        assert classconfs.size() == frequency.size()
        class_ece[i] = ece_loss(classconfs, frequency, bins, norm)

    return torch.mean(class_ece).item()

def cal_kde(logits, labels, idx_test, norm=1, kbw_choice='correct'):
    logits, labels = logits[idx_test], labels[idx_test]
    confidence, preds = torch.softmax(logits, -1).max(dim=-1)
    corrects = (preds == labels)

    # ece_kde(confs, corrects, norm=norm)
    return KDE.ece_kde(confidence, corrects, norm=norm).item()

def cal_ece2(logits, labels, idx_test, n_bins=15):

    logits, labels = logits[idx_test], labels[idx_test]
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

class KDE: 
    """
    Code adapted from https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    """
    @staticmethod
    def mirror_1d(d, xmin=None, xmax=None):
        """If necessary apply reflecting boundary conditions."""
        if xmin is not None and xmax is not None:
            xmed = (xmin+xmax)/2
            return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
        elif xmin is not None:
            return np.concatenate((2*xmin-d, d))
        elif xmax is not None:
            return np.concatenate((d, 2*xmax-d))
        else:
            return d

    @staticmethod
    def density_estimator(conf, x_int, kbw, method='triweight'):
        # Compute KDE using the bandwidth found, and twice as many grid points
        low_bound, up_bound = 0.0, 1.0
        pp = FFTKDE(bw=kbw, kernel=method).fit(conf).evaluate(x_int)
        pp[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        return pp * 2  # Double the y-values to get integral of ~1       
        
    @staticmethod
    @torch.no_grad()
    def ece_kde(confidence, correct, norm=1, kbw_choice='correct'):
        confidence = torch.clip(confidence,1e-256,1-1e-256)
        x_int = np.linspace(-0.6, 1.6, num=2**14)
        correct_conf = (confidence[correct==1].view(-1,1)).cpu().numpy()
        N = confidence.size(0)

        if kbw_choice == 'correct':
            kbw = np.std(correct_conf)*(N*2)**-0.2
        else:
            kbw = np.std(confidence.cpu().numpy())*(N*2)**-0.2
        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = KDE.mirror_1d(correct_conf,low_bound,up_bound)
        pp1 = KDE.density_estimator(dconf_1m, x_int, kbw)
        pp1 = torch.from_numpy(pp1).to(confidence.device)

        pred_b_intm = KDE.mirror_1d(confidence.view(-1,1).cpu().numpy(),low_bound,up_bound)
        pp2 = KDE.density_estimator(pred_b_intm, x_int, kbw)
        pp2 = torch.from_numpy(pp2).to(confidence.device)

        # Accuracy (confidence)
        perc = torch.mean(correct.float())
        x_int = torch.from_numpy(x_int).to(confidence.device)
        integral = torch.zeros_like(x_int)

        conf = x_int
        accu = perc*pp1/pp2
        accu = torch.where((accu < 1.0), accu ,1.0)
        thre = ( pp1 > 1e-6) | (pp2 > 1e-6 ) 
        accu_notnan = ~torch.isnan(accu)
        integral[thre & accu_notnan] = torch.abs(conf[thre & accu_notnan]-accu[thre & accu_notnan])**norm*pp2[thre & accu_notnan]
        # Dont integrate the first sample 
        fail_thre_index = torch.nonzero(~thre)[1:]
        integral[fail_thre_index] = integral[fail_thre_index-1]

        ind = (x_int >= 0.0) & (x_int <= 1.0)
        return torch.trapz(integral[ind],x_int[ind]) / torch.trapz(pp2[ind],x_int[ind])


def partial_sums(t, lens):
    device = t.device
    elems, parts = t.size(0), len(lens)
    ind_x = torch.repeat_interleave(torch.arange(parts, device=device), lens)
    total = len(ind_x)
    ind_mat = torch.sparse_coo_tensor(
        torch.stack((ind_x, torch.arange(total, device=device)), dim=0),
        torch.ones(total, device=device, dtype=t.dtype),
        (parts, elems),
        device=device)
    return torch.mv(ind_mat, t)

class Reliability(NamedTuple):
    conf: Tensor
    acc: Tensor
    count: LongTensor

def set_global_seeds(seed):
    """
    Set global seed for reproducibility
    """  
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)

def arg_parse(add_method=None):
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora','Citeseer', 'Pubmed', 
                        'Computers', 'Photo', 'CS', 'Physics', 'CoraFull'])
    parser.add_argument('--split_type', type=str, default='5_3f_85', help='k-fold and test split')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--verbose', action='store_true', default=False, help='Show training and validation loss')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    parser.add_argument('--calibration', type=str, default='GATS',  help='Post-hoc calibrators')
    parser.add_argument('--cal_wdecay', type=float, default=None, help='Weight decay for calibration phase')
    parser.add_argument('--cal_dropout_rate', type=float, default=0.5, help='Dropout rate for calibrators (CaGCN)')
    parser.add_argument('--folds', type=int, default=3, help='K folds cross-validation for calibration')
    parser.add_argument('--ece-bins', type=int, default=15, help='number of bins for ece')
    parser.add_argument('--ece-scheme', type=str, default='equal_width', help='binning scheme for ece')
    parser.add_argument('--ece-norm', type=float, default=1.0, help='norm for ece')
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--config', action='store_true', default=False)

    gats_parser = parser.add_argument_group('optional GATS arguments')
    gats_parser.add_argument('--heads', type=int, default=2, help='Number of heads for GATS. Hyperparameter set: {1,2,4,8,16}')
    gats_parser.add_argument('--bias', type=float, default=1, help='Bias initialization for GATS')
    
    if add_method:
        add_method(parser)
    
    args = parser.parse_args()
    if args.config:
        config = read_config(args)
        for key, value in config.items():
            setattr(args, key, value)

    args_dict = {}
    for group in parser._action_groups:
        if group.title == 'optional GATS arguments':
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict['gats_args'] = argparse.Namespace(**group_dict)
        else:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict.update(group_dict)
    return argparse.Namespace(**args_dict)

def read_config(args):
    dir = Path(os.path.join('config', args.calibration))
    file_name = f'{args.dataset}_{args.model}.yaml'
    try:
        with open(dir/file_name) as file:
            yaml_file = yaml.safe_load(file)
    except IOError:
        yaml_file = {}
    if yaml_file is None:
        yaml_file = {}
    return yaml_file

def default_cal_wdecay(args):
    if args.calibration in ['TS', 'VS', 'ETS']:
        return 0
    elif args.calibration == 'CaGCN':
        if args.dataset == "CoraFull":
            return 0.03
        else:
            return 5e-3
    else:
        return 5e-4

def name_model(fold, args):
    assert args.model in ['GCN', 'GAT'], f'Unexpected model name {args.model}.'
    name = args.model
    name += "_dp" + str(args.dropout_rate).replace(".","_") + "_"
    try:
        power =-math.floor(math.log10(args.wdecay))
        frac = str(args.wdecay)[-1] if power <= 4 else str(args.wdecay)[0]
        name += frac + "e_" + str(power)
    except:
        name += "0"
    name += "_f" + str(fold)
    return name

def metric_mean(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'cls_ece', 'kde']:
            weight = 100
        else:
            weight = 1
        out[key] = np.mean(val) * weight
    return out

def metric_std(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'cls_ece', 'kde']:
            weight = 100
        else:
            weight = 1
        out[key] = np.sqrt(np.var(val)) * weight
    return out

def create_nested_defaultdict(key_list):
    # To do: extend to *args
    out = {}
    for key in key_list:
        out[key] = defaultdict(list)
    return out

def save_prediction(predictions, name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    np.save(raw_dir/file_name, predictions)

def load_prediction(name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    return np.load(raw_dir / file_name)

def plot_reliabilities(
        reliabilities: Sequence[Reliability], title, saveto, bgcolor='w'):
    linewidth = 1.0

    confs = [(r[0] / (r[2] + torch.finfo().tiny)).cpu().numpy()
             for r in reliabilities]
    accs = [(r[1] / (r[2] + torch.finfo().tiny)).cpu().numpy()
            for r in reliabilities]
    masks = [r[2].cpu().numpy() > 0 for r in reliabilities]

    nonzero_counts = np.sum(np.asarray(masks, dtype=np.long), axis=0)
    conf_mean = np.sum(
        np.asarray(confs), axis=0) / (nonzero_counts + np.finfo(np.float).tiny)
    acc_mean = np.sum(
        np.asarray(accs), axis=0) / (nonzero_counts + np.finfo(np.float).tiny)
    acc_std = np.sqrt(
        np.sum(np.asarray(accs) ** 2, axis=0)
        / (nonzero_counts + np.finfo(np.float).tiny)
        - acc_mean ** 2)
    conf_mean = conf_mean[nonzero_counts > 0]
    acc_mean = acc_mean[nonzero_counts > 0]
    acc_std = acc_std[nonzero_counts > 0]

    fig, ax1 = plt.subplots(figsize=(2, 2), facecolor=bgcolor)
    for conf, acc, mask in zip(confs, accs, masks):
        ax1.plot(
            conf[mask], acc[mask], color='lightgray',
            linewidth=linewidth / 2.0, zorder=0.0)
    ax1.plot(
        [0, 1], [0, 1], color='black', linestyle=':', linewidth=linewidth,
        zorder=0.8)
    ax1.plot(
        conf_mean, acc_mean, color='blue', linewidth=linewidth, zorder=1.0)
    ax1.fill_between(
        conf_mean, acc_mean - acc_std, acc_mean + acc_std, facecolor='b',
        alpha=0.3, zorder=0.9)

    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    # ax1.legend(loc="lower right")
    ax1.set_title(title)
    plt.tight_layout()
    ax1.set_aspect(1)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(saveto, bbox_inches='tight', pad_inches=0)