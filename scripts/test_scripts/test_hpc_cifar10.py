import torch
import torch.nn as nn
import torchvision

import argparse
import sys
import os

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers

class Custom_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential()
        self.seq.add_module('conv1', nn.Conv2d(3, 16, 3))
        self.seq.add_module('act1', nn.ReLU())
        self.seq.add_module('conv2', nn.Conv2d(16, 32, 3))
        self.seq.add_module('act2', nn.ReLU())
        self.seq.add_module('conv3', nn.Conv2d(32, 64, 3))
        self.seq.add_module('act3', nn.ReLU())
        self.seq.add_module('pool', nn.AdaptiveAvgPool2d(1))

        self.head = nn.Sequential()
        self.head.add_module('lin1', nn.Linear(64, 64))
        self.head.add_module('lin1_actv', nn.ReLU())
        self.head.add_module('clf', nn.Linear(64, 10))
    
    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class PL_Model(pl.LightningModule):
    def __init__(self, pid=None, queue=None):
        super().__init__()

        self.model = Custom_Model()
        self.criteria = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.queue = queue
        self.pid = pid

    def forward(self, x):
        return self.model(x)

    @pl.data_loader
    def train_dataloader(self):
        train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=self.transform)
        train_data.data = train_data.data[:512]
        train_data.targets = train_data.targets[:512]
        data_loader = torch.utils.data.DataLoader(train_data,
            batch_size=8, 
            shuffle=True)
        return data_loader
    
    @pl.data_loader
    def val_dataloader(self):
        val_data = torchvision.datasets.CIFAR10('./data', train=False, transform=self.transform)
        val_data.data = val_data.data[:512]
        val_data.targets = val_data.targets[:512]
        data_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
        return data_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criteria(logit, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criteria(logit, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        if self.queue is not None:
            self.queue.put([self.pid, self.trainer.current_epoch])
        return {}

def worker(pid, queue):
    model = PL_Model(pid, queue)
    tb_logger = pl_loggers.TensorBoardLogger('logs_test', name='test'+str(pid), version='')
    trainer = Trainer(
        logger=tb_logger,
        max_epochs=10, 
        num_sanity_val_steps=0, 
        weights_summary=None,
        progress_bar_refresh_rate=0,
        gpus=1
        )
    trainer.fit(model)


def main():
    processes_nb = 4
    progress_bars = []
    processes = []
    progress_queue = mp.Queue()
    for _ in range(processes_nb):
        progress_bars.append(tqdm(total=10))
    for i in range(processes_nb):
        proc = mp.Process(target=worker, args=(i, progress_queue,))
        processes.append(proc)
        proc.start()
    while any([proc.is_alive() for proc in processes]):
        pid, status = progress_queue.get()
        if status:
            progress_bars[pid].update()
        if status == 9:
            processes[pid].join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    LOGDIR = 'logs_test'
    os.makedirs('./'+LOGDIR, exist_ok=True)
    # torchvision.datasets.CIFAR10('./data', train=True, download=True)
    # torchvision.datasets.CIFAR10('./data', train=False, download=True)
    main()



