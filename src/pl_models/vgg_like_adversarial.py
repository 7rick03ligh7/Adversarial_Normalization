import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl

from src.models.ad_model import Adversarial_VGGLike
from src.utils.utils import (calc_confusion_mtx, visualize_confustion_matrix,
                             save_model, set_seeds, conv2d_weight_init, linear_weight_init)

class VGGLike_Adversarial(pl.LightningModule):
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
        self.model = Adversarial_VGGLike(self.model_params)

        set_seeds(self.model_params['seed'])
        self.model.conv1.apply(conv2d_weight_init)
        self.model.conv2.apply(conv2d_weight_init)
        self.model.conv3.apply(conv2d_weight_init)
        self.model.conv4.apply(conv2d_weight_init)
        self.model.conv5.apply(conv2d_weight_init)
        self.model.conv6.apply(conv2d_weight_init)
        self.model.clf.apply(linear_weight_init)

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
            )
        return data_loader

    def calc_loss(self, logits, target):
        loss = F.cross_entropy(logits, target)
        return loss

    def calc_advloss(self, reals, fakes):
        # adv_loss = torch.zeros(1, requires_grad=True).cuda()
        # for real, fake in zip(reals, fakes):
        #     adv_loss += (real - fake).mean()
        # return adv_loss
        return fakes.mean() - reals.mean()

    def configure_optimizers(self):
        main_params =   list(self.model.conv1.parameters()) + \
                        list(self.model.conv2.parameters()) + \
                        list(self.model.conv3.parameters()) + \
                        list(self.model.conv4.parameters()) + \
                        list(self.model.conv5.parameters()) + \
                        list(self.model.conv6.parameters()) + \
                        list(self.model.clf.parameters())
        optimizer_ce  = optim.Adam(
            main_params,
            lr=5e-4,
            weight_decay=self.model_params['weight_decay']
        )

        # adversarial_params =   list(self.model.adv_norm1.parameters()) + \
        #             list(self.model.adv_norm2.parameters()) + \
        #             list(self.model.adv_norm3.parameters()) + \
        #             list(self.model.adv_norm4.parameters())
        adversarial_params =   list(self.model.adv_norm1.parameters())
        optimizer_discriminator = optim.Adam(
            adversarial_params,
            lr=5e-4,
            weight_decay=self.model_params['weight_decay']
        )

        generator_params =  list(self.model.conv1.parameters()) + \
                            list(self.model.conv2.parameters()) + \
                            list(self.model.conv3.parameters()) + \
                            list(self.model.conv4.parameters())
        optimizer_generator = optim.Adam(
            generator_params,
            lr=5e-4,
            weight_decay=self.model_params['weight_decay']
        )

        return (
            {'optimizer': optimizer_ce, 'frequency': 1},
            # {'oprimizer': optimizer_generator, 'frequency': 1},
            {'optimizer': optimizer_discriminator, 'frequency': 4}
        )

        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, using_native_amp, using_lbfgs):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 1:
            torch.nn.utils.clip_grad_norm_(self.model.adv_norm1.parameters(), 0.01)
            optimizer.step()
            optimizer.zero_grad()

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, target = batch

        if optimizer_idx == 0:
            logits = self.model(image, typeof_forward = 0)
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

        if optimizer_idx == 1:
            z_real = self.model(image, typeof_forward = 1)
            # y_real = torch.ones(image.shape[0]) * -1

            gauss = torch.randn(image.shape, requires_grad=True).cuda()
            z_gauss = self.model(gauss, typeof_forward = 1)
            # y_fake = torch.ones(image.shape[0])

            loss = self.calc_advloss(z_real[0], z_gauss[0])

            outputs = {
                'loss': loss
            }
            return outputs

    def training_epoch_end(self, outputs):
        print(self.trainer.current_epoch)

        first_opt1 = 0
        for i in range(5):
            if 'metrics' in outputs[i].keys():
                first_opt1 = i

        outputs = outputs[first_opt1::5]
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
        logits = self.model(image, typeof_forward = 0)
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

        
        self.queue.put([self.pid, self.trainer.current_epoch])
        # if self.trainer.current_epoch%50 == 0:
        #     print(self.trainer.current_epoch)

        for metric in outputs[0]['metrics'].keys():
            metric_values = torch.stack([x['metrics'][metric] for x in outputs])
            avg_metric = metric_values.mean()
            self.history['val'][metric].append(avg_metric.item())

        print('epoch end')
        return {
            'val_loss': avg_loss,
            'val_acc': torch.FloatTensor([self.history['val']['acc'][-1]])
        }
