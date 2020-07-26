import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import confusion_matrix

from BN_LN_IN import BN_LN_IN_VGGLike
from WN import WN_VGGLike


def conv2dWeightInit(x):
    if isinstance(x, nn.Conv2d):
        torch.nn.init.xavier_normal_(x.weight)
        torch.nn.init.zeros_(x.bias)

def linearWeightInit(x):
    if isinstance(x, nn.Linear):
        torch.nn.init.xavier_normal_(x.weight)
        torch.nn.init.zeros_(x.bias)

class VGGLike(pl.LightningModule):
    def __init__(self, model_params, logpath='.', seed=42):
        super().__init__()
        self.history = {
            'train_loss': [],
            'train': {'loss':[], 'acc':[]},
            'val': {'loss':[], 'acc':[]}
        }

        self.seed = seed
        self.model_params = model_params
        self.model = None
        self.configure_model()
        self.transform = transforms.Compose([
            transforms.ToTensor()])

        self.targets = []
        self.predicts = []
        self.confusion_matrix = torch.zeros(self.model_params['classes_nb'],
                                            self.model_params['classes_nb'], dtype=int)
        self.confusion_matrix_norm = torch.zeros(self.model_params['classes_nb'],
                                            self.model_params['classes_nb'], dtype=int)
        self.logpath = logpath
        self.vis_dir = 'conf_matrix'
        self.ckpt_every_dir = 'every'

        os.makedirs(self.logpath+'/'+self.vis_dir, exist_ok=True)
        os.makedirs(self.logpath+'/checkpoints/'+self.ckpt_every_dir, exist_ok=True)

    def setSeeds(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all((self.seed+1)*2)
        np.random.seed((self.seed+2)*3)
        random.seed((self.seed+3)*4)

    def configure_model(self):
        if self.model_params['regulz_type'] in (
            'BatchNorm', 
            'LayerNorm', 
            'SpLayerNorm', 
            'InstanceNorm', 
            'None'
            ):
            self.model = BN_LN_IN_VGGLike(self.model_params)
        if self.model_params['regulz_type'] in ('WeightNorm'):
            self.model = WN_VGGLike(self.model_params)

        self.setSeeds()
        self.model.apply(conv2dWeightInit)
        self.model.apply(linearWeightInit)

    def forward(self, x):
        return self.model(x)

    @pl.data_loader
    def train_dataloader(self):
        train_data = torchvision.datasets.CIFAR10(
            './data', 
            train=True, 
            transform=self.transform, 
            download=False
            )
        data_loader = DataLoader(
            train_data,
            batch_size=self.model_params['batch_size'],
            shuffle=True,
            num_workers=4
            )
        if self.seed:
            self.setSeeds()
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        val_data = torchvision.datasets.CIFAR10(
            './data',
            train=False,
            transform=self.transform,
            download=False
            )
        data_loader = DataLoader(
            val_data,
            batch_size=self.model_params['batch_size'],
            num_workers=4
            )
        return data_loader

    def calc_loss(self, logits, target):
        loss = F.cross_entropy(logits, target)
        return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=5e-4,
            weight_decay=self.model_params['weight_decay']
        )

    def calc_confusion_matrix(self):
        targets = torch.cat(self.targets)
        preds = torch.cat(self.predicts)
        self.confusion_matrix = confusion_matrix(preds, targets).detach().cpu()

    def visualize_confustion_matrix(self, normalize=True):
        if normalize:
            self.confusion_matrix_norm = self.confusion_matrix/self.confusion_matrix.sum(dim=1)
        fig, ax = plt.subplots(figsize=(8,6), dpi=200)
        sns.heatmap(self.confusion_matrix_norm, ax=ax, cmap='Blues', vmin=0, vmax=1, annot=self.confusion_matrix.int(), fmt='d')
        ax.set_xlabel('PREDICTED', fontsize=14)
        ax.set_ylabel('GROUND TRUTH', fontsize=14)
        ax.set_title(f'Epoch_{self.trainer.current_epoch}', fontsize=14)
        ax.set_xticklabels(self.model_params['classes'], rotation=65, fontsize=12)
        ax.set_yticklabels(self.model_params['classes'], rotation=0, fontsize=12)
        fig.savefig(self.logpath+'/'+self.vis_dir+f'/epoch_{self.trainer.current_epoch}.png', bbox_inches="tight")
        plt.close('all')

    def save_model(self):
        path = self.trainer.logger.log_dir.split(os.path.sep)
        path.append('checkpoints')
        path.append(self.ckpt_every_dir)
        path = os.path.sep.join(path)
        self.trainer.save_checkpoint(path + f'/epoch={self.trainer.current_epoch}.ckpt')

    def training_step(self, batch, batch_idx):
        image, target = batch
        logits = self.model(image)
        loss = self.calc_loss(logits, target)
        metrics = {
            'acc': (logits.max(dim=1)[1] == target).sum().float()/len(target)
        }
        outputs = {
            'loss': loss,
            'metrics': metrics
        }
        self.history['train_loss'].append(loss.item())
        return outputs

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs])
        avg_loss = loss.mean()
        self.history['train']['loss'].append(avg_loss.item())

        for metric in outputs[0]['metrics'].keys():
            metric_values = torch.stack([x['metrics'][metric] for x in outputs])
            avg_metric = metric_values.mean()
            self.history['train'][metric].append(avg_metric.item())
        return {
            'train_loss': avg_loss
        }

    def validation_step(self, batch, batch_idx):
        image, target = batch
        logits = self.model(image)
        loss = self.calc_loss(logits, target)
        class_preds = logits.max(dim=1)[1]

        self.predicts.append(class_preds)
        self.targets.append(target)

        metrics = {
            'acc': (class_preds == target).sum().float()/len(target)
        }
        outputs = {
            'loss': loss,
            'metrics': metrics
        }
        return outputs

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs])
        avg_loss = loss.mean()
        self.history['val']['loss'].append(avg_loss.item())

        self.save_model()
        self.calc_confusion_matrix()
        self.visualize_confustion_matrix(normalize=True)
        self.targets = []
        self.predicts = []

        if self.trainer.current_epoch%50 == 0:
            print(self.trainer.current_epoch)

        for metric in outputs[0]['metrics'].keys():
            metric_values = torch.stack([x['metrics'][metric] for x in outputs])
            avg_metric = metric_values.mean()
            self.history['val'][metric].append(avg_metric.item())
        return {
            'val_loss': avg_loss,
            'val_acc': torch.FloatTensor([self.history['val']['acc'][-1]])
        }
