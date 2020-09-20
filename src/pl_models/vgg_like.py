import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl

from src.models.bn_ln_in_model import BN_LN_IN_VGGLike
from src.models.wn_model import WN_VGGLike
from src.models.sn_model import SN_VGGLike

from src.utils.utils import (calc_confusion_mtx, visualize_confustion_matrix,
                             save_model, set_seeds, conv2d_weight_init, 
                             linear_weight_init, sn_weight_init)

class VGGLike(pl.LightningModule):
    def __init__(self, pid, queue, model_params):
        super().__init__()
        self.history = {
            'train_loss': [],
            'train': {'loss':[], 'acc':[]},
            'val': {'loss':[], 'acc':[]}
        }

        self.queue = queue
        self.pid = pid
        self.model_params = model_params
        self.model = None
        self.configure_model()

        if model_params['regulz_type'] == 'SelfNorm':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0,1)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()])

        self.targets = []
        self.predicts = []
        self.confusion_mtx = torch.zeros(
            self.model_params['classes_nb'],
            self.model_params['classes_nb'], dtype=int
            )
        self.vis_dir = 'conf_matrix'
        self.ckpt_every_dir = 'every'

        os.makedirs(self.model_params['logpath'] + '/'+self.vis_dir, exist_ok=True)
        os.makedirs(self.model_params['logpath'] + '/checkpoints/'+self.ckpt_every_dir, exist_ok=True)


    def configure_model(self):
        set_seeds(self.model_params['seed'])

        if self.model_params['regulz_type'] in (
                'BatchNorm',
                'LayerNorm',
                'SpLayerNorm',
                'InstanceNorm',
                'None'
            ):
            self.model = BN_LN_IN_VGGLike(self.model_params)
            self.model.apply(conv2d_weight_init)
            self.model.apply(linear_weight_init)

        if self.model_params['regulz_type'] == 'WeightNorm':
            self.model = WN_VGGLike(self.model_params)
            self.model.apply(conv2d_weight_init)
            self.model.apply(linear_weight_init)

        if self.model_params['regulz_type'] == 'SelfNorm':
            self.model = SN_VGGLike(self.model_params)
            self.model.apply(conv2d_weight_init)
            self.model.apply(linear_weight_init)   

            # reinitialize conv layer weights with selu activation
            self.model.conv.conv_0.apply(sn_weight_init)
            self.model.conv.conv_1.apply(sn_weight_init)
            self.model.conv.conv_3.apply(sn_weight_init)
            self.model.conv.conv_4.apply(sn_weight_init)

    def forward(self, x):
        return self.model(x)

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
        set_seeds(self.model_params['seed'])
        return data_loader

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

        # return optim.Adam([
        #     {'params': self.model.conv.conv_0.parameters(), 'weight_decay': 0},
        #     {'params': self.model.conv.conv_1.parameters(), 'weight_decay': 0},
        #     {'params': self.model.conv.conv_3.parameters(), 'weight_decay': 0},
        #     {'params': self.model.conv.conv_4.parameters(), 'weight_decay': 0},
        # ], lr=5e-4, weight_decay=self.model_params['weight_decay']
        # )

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

        save_model(self.trainer, self.ckpt_every_dir)
        self.confusion_mtx = calc_confusion_mtx(self.targets, self.predicts)
        visualize_confustion_matrix(self.confusion_mtx, self.trainer, self.model_params, self.vis_dir, normalize=True)
        self.targets = []
        self.predicts = []

        # print(f'pid={self.pid}, epoch={self.current_epoch}')
        self.queue.put([self.pid, self.trainer.current_epoch])
        # if self.trainer.current_epoch%50 == 0:
        #     print(self.trainer.current_epoch)

        for metric in outputs[0]['metrics'].keys():
            metric_values = torch.stack([x['metrics'][metric] for x in outputs])
            avg_metric = metric_values.mean()
            self.history['val'][metric].append(avg_metric.item())
        return {
            'val_loss': avg_loss,
            'val_acc': torch.FloatTensor([self.history['val']['acc'][-1]])
        }
