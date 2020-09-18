import os
import numpy as np

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
        self.adv_history = {
            'generator_loss': [],
            'adversarial_loss': []
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
        set_seeds(self.model_params['seed'])
        
        self.model = Adversarial_VGGLike(self.model_params)
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

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        interpolates = []
        for i in range(len(fake_samples)):
            # Random weight term for interpolation between real and fake samples
            alpha = torch.Tensor(np.random.random((real_samples[i].size(0), 1, 1, 1))).cuda()
            # Get random interpolation between real and fake samples
            interpolates.append((alpha * real_samples[i] + ((1 - alpha) * fake_samples[i])).requires_grad_(True))
        
        gradient_penalty = []
        d_interpolates_list = self.model(interpolates, typeof_forward = 2)
        for i, d_interpolates in enumerate(d_interpolates_list):
            fake = torch.autograd.Variable(torch.Tensor(real_samples[i].shape[0], 1).fill_(1.0), requires_grad=False).cuda()
            
            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates[i],
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty.append(((gradients.norm(2, dim=1) - 1) ** 2).mean())

        gradient_penalty = torch.stack(gradient_penalty)
        gradient_penalty = gradient_penalty.mean()
        return gradient_penalty

    def calc_advloss(self, reals, fakes, penalty, lambda_gp = 20):
        loss = torch.stack([torch.mean(fakes[i]) - torch.mean(reals[i]) for i in range(len(reals))])
        loss = torch.mean(loss) + lambda_gp*penalty
        return loss

    def calc_genloss(self, logits):
        loss = torch.stack([-torch.mean(logit) for logit in logits])
        loss = torch.mean(loss)
        return loss

    def configure_optimizers(self):
        main_params =   list(self.model.conv1.parameters()) + \
                        list(self.model.conv2.parameters()) + \
                        list(self.model.conv3.parameters()) + \
                        list(self.model.conv4.parameters()) + \
                        list(self.model.conv5.parameters()) + \
                        list(self.model.conv6.parameters()) + \
                        list(self.model.clf.parameters())
        optimizer_ce =  optim.Adam(
            main_params,
            lr=5e-5,
            betas=[0, 0.9],
            weight_decay=self.model_params['weight_decay']
        )

        generator_params =  list(self.model.conv1.parameters()) + \
                            list(self.model.conv2.parameters()) + \
                            list(self.model.conv3.parameters()) + \
                            list(self.model.conv4.parameters())
        # generator_params =  list(self.model.conv1.parameters())
        optimizer_generator = optim.Adam(
            generator_params,
            lr=5e-5,
            betas=[0, 0.9],
            weight_decay=self.model_params['weight_decay']
        )

        adversarial_params =    list(self.model.adv_norm1.parameters()) + \
                                list(self.model.adv_norm2.parameters()) + \
                                list(self.model.adv_norm3.parameters()) + \
                                list(self.model.adv_norm4.parameters())
        # adversarial_params =   list(self.model.adv_norm1.parameters())
        optimizer_discriminator = optim.Adam(
            adversarial_params,
            lr=5e-5,
            betas=[0, 0.9],
            weight_decay=self.model_params['weight_decay']
        )

        return (
            {'optimizer': optimizer_ce,             'frequency': 1},
            {'optimizer': optimizer_generator,      'frequency': 1},
            {'optimizer': optimizer_discriminator,  'frequency': 8}
        )
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, using_native_amp, using_lbfgs):

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 2:
            optimizer.step()
            optimizer.zero_grad()

            # for p in self.model.adv_norm1.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            # for p in self.model.adv_norm2.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            # for p in self.model.adv_norm3.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            # for p in self.model.adv_norm4.parameters():
            #     p.data.clamp_(-0.01, 0.01)


    def training_step(self, batch, batch_idx, optimizer_idx):
        image, target = batch

        # Cross entropy
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

        # Generator loss
        if optimizer_idx == 1:
            gauss = torch.randn(image.shape, requires_grad=True).cuda()
            z_gauss = self.model(gauss, typeof_forward = 1)
            loss = self.calc_genloss(z_gauss)
            
            self.adv_history['generator_loss'].append(loss.item())
            outputs = {
                'loss': loss
            }
            return outputs

        # Adversarial loss
        if optimizer_idx == 2:
            z_real = self.model(image, typeof_forward = 1)          # generator
            advers_real = self.model(z_real, typeof_forward = 2)    # discriminator
            # y_real = torch.ones(image.shape[0]) * -1

            gauss = torch.randn(image.shape, requires_grad=True).cuda()
            z_gauss = self.model(gauss, typeof_forward = 1)         # generator
            advers_gauss =self.model(z_gauss, typeof_forward = 2)   # discriminator

            # y_fake = torch.ones(image.shape[0])

            penalty = self.compute_gradient_penalty(z_real, z_gauss)
            # penalty = 0
            loss = self.calc_advloss(advers_real, advers_gauss, penalty)
            self.adv_history['adversarial_loss'].append(loss.item())

            outputs = {
                'loss': loss
            }
            return outputs

    def training_epoch_end(self, outputs):
        print(self.trainer.current_epoch)

        ce_opt_index = 0
        opt_step = 10
        for i in range(opt_step):
            if 'metrics' in outputs[i].keys():
                ce_opt_index = i

        outputs = outputs[ce_opt_index::opt_step]
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
