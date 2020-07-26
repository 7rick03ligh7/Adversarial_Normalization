import os
import sys
import random
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.classification import confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint

LOGDIR = 'logs'
os.makedirs('./'+LOGDIR, exist_ok=True)
torchvision.datasets.CIFAR10('./data', train=True, download=True)
torchvision.datasets.CIFAR10('./data', train=False, download=True)