import os
import random
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch import autograd
from pytorch_lightning.metrics.classification import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calc_confusion_mtx(targets: list, predicts: list):
    targets = torch.cat(targets)
    preds = torch.cat(predicts)
    confusion_mtx = confusion_matrix(preds, targets).detach().cpu()

    return confusion_mtx

def visualize_confustion_matrix(confusion_mtx, trainer, model_params, vis_dir, normalize=True):
    if normalize:
        confusion_mtx_norm = confusion_mtx/confusion_mtx.sum(dim=1)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    sns.heatmap(
        confusion_mtx_norm,
        ax=ax, cmap='Blues',
        vmin=0,
        vmax=1,
        annot=confusion_mtx.int(),
        fmt='d'
        )
    ax.set_xlabel('PREDICTED', fontsize=14)
    ax.set_ylabel('GROUND TRUTH', fontsize=14)
    ax.set_title(f'Epoch_{trainer.current_epoch}', fontsize=14)
    ax.set_xticklabels(model_params['classes'], rotation=65, fontsize=12)
    ax.set_yticklabels(model_params['classes'], rotation=0, fontsize=12)
    fig.savefig(
        model_params['logpath'] + '/' + vis_dir + f'/epoch_{trainer.current_epoch}.png',
        bbox_inches="tight"
        )
    plt.close('all')

def save_model(trainer, ckpt_every_dir):
    path = trainer.logger.log_dir.split(os.path.sep)
    path.append('checkpoints')
    path.append(ckpt_every_dir)
    path = os.path.sep.join(path)
    trainer.save_checkpoint(path + f'/epoch={trainer.current_epoch}.ckpt')


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all((seed+1)*2)
    np.random.seed((seed+2)*3)
    random.seed((seed+3)*4)

def conv2d_weight_init(x):
    if isinstance(x, nn.Conv2d):
        torch.nn.init.xavier_normal_(x.weight)
        torch.nn.init.zeros_(x.bias)

def linear_weight_init(x):
    if isinstance(x, nn.Linear):
        torch.nn.init.xavier_normal_(x.weight)
        torch.nn.init.zeros_(x.bias)

def sn_weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        m.weight.data.normal_(0, np.sqrt(1. /n))
        if hasattr(m.bias, 'data'):
            m.bias.data.zero_()
    if isinstance(m, nn.Linear):
        n = m.in_features * m.out_features
        m.weight.data.normal_(0, np.sqrt(1. /n))
        if hasattr(m.bias, 'data'):
            m.bias.data.zero_()

def generate_filters(conv_layers_nb, filters_base, begin_factor=2):
    filters = [begin_factor*filters_base]
    interv = 2
    difference = 1
    factor = begin_factor
    for i in range(conv_layers_nb-1):
        if (i+1)%interv == 0 and i != 0:
            difference *= 2
        factor += difference
        filters.append(factor*filters_base)
    return filters

def generate_logname(model_params):
    logname = ''
    logname += 'fnb_' + str(len(model_params['filters'])) + '--'
    logname += 'fbase_' + str(int(model_params['filters'][0]/2)) + '--'
    logname += 'wdecay_' + str(model_params['weight_decay']) + '--'
    logname += 'elu_' + str(model_params['elu_alpha']) + '--'
    if model_params['regulz_type'] == 'WeightNorm':
        rname = 'WN_'
    if model_params['regulz_type'] == 'BatchNorm':
        rname = 'BN_' + str(model_params['batch_size'])
    if model_params['regulz_type'] == 'InstanceNorm':
        rname = 'IBN_'
    if model_params['regulz_type'] == 'LayerNorm':
        rname = 'LN_'
    if model_params['regulz_type'] == 'SpLayerNorm':
        rname = 'SLN_'
    if model_params['regulz_type'] == 'SelfNorm':
        rname = 'SN_'
    if model_params['regulz_type'] == 'AdversarialNorm':
        rname = 'AN-'
        if model_params['advers_type'] == 'Layer':
            rname += model_params['advers_type'] + '_'
        rname += str(model_params['advers_f_num'])
    if model_params['regulz_type'] == 'None':
        rname = 'NONE_'
    logname += rname
    return logname