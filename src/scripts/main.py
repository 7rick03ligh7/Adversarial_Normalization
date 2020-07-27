import os
import sys
import random
import shutil

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models.vgg_Like import VGGLike
import multiprocessing as mp
import argparse
import json
from src.utils.generates import *


def worker(pid, queue, model_params):
    model = VGGLike(model_params)

def main(models_params):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--params_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42, required=False)
    arguments = parser.parse_args()

    with open(arguments.params_file, 'r') as f:
        models_params = json.load(f)

    for model_params in models_params:
        model_params['classes'] = [
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        ]
        logname, logpath = generate_logname(model_params)
        model_params['logpath'] = logpath
        model_params['seed'] = arguments.seed

