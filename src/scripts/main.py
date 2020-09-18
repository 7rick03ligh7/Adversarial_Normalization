import os
import sys
import shutil
sys.path.append(os.path.sep.join(sys.path[0].split(os.path.sep)[:-2]))

import pandas as pd
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.pl_models.vgg_like import VGGLike
from src.pl_models.vgg_like_adversarial import VGGLike_Adversarial
import multiprocessing as mp
import argparse
import json
from src.utils.utils import generate_logname
from tqdm import tqdm
import time

import logging
logging.getLogger('lightning').setLevel(0)

import warnings
warnings.filterwarnings('ignore')


def worker(pid, queue, model_params):
    # if os.path.exists(model_params['logpath']):
    #     shutil.rmtree(model_params['logpath'])
    # os.makedirs(model_params['logpath'], exist_ok=True)

    if model_params['adversarial']:
        model = VGGLike_Adversarial(0, mp.Queue(), model_params)
    else:
        model = VGGLike(pid, queue, model_params)
    tb_logger = pl_loggers.TensorBoardLogger(
        model_params['logdir'],
        name=model_params['logname'],
        version=''
        )
    os.makedirs(model_params['logpath']+'/checkpoints/best/', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_params['logpath']+'/checkpoints/best/',
        save_top_k=1,
        verbose=False,
        monitor='val_acc',
        mode='max',
        prefix=''
    )

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        gpus = 1
        print('!!!!!!!! WARNING WARNING WARNING WARNING !!!!!!!!')
        print('!!!!!!!! CUDA_VISIBLE_DEVICES IS NOT EXIST !!!!!!!!')
    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=model_params['epochs'],
        num_sanity_val_steps=0,
        weights_summary=None,
        progress_bar_refresh_rate=0,
        track_grad_norm=1,
        gpus=gpus
        )
    trainer.fit(model)

    pd.DataFrame(model.history['train_loss'], columns=['train_loss']).to_csv(model_params['logpath']+'/train_loss.csv', index=False)
    pd.DataFrame(model.history['train']).to_csv(model_params['logpath']+'/train.csv', index=False)
    pd.DataFrame(model.history['val']).to_csv(model_params['logpath']+'/val.csv', index=False)

def main_parallel(models_params):
    progress_bars = []
    processes = []
    progress_queue = mp.Queue()
    for i, model_params in enumerate(models_params):
        proc = mp.Process(target=worker, args=(i, progress_queue, model_params,))
        processes.append(proc)
        proc.start()

    time.sleep(5)       # crutch for printing progress bar after all debug printings
    for i, _ in enumerate(models_params):
        progress_bars.append(tqdm(total=model_params['epochs']))

    while any([proc.is_alive() for proc in processes]):
        pid, status = progress_queue.get()
        if status:
            progress_bars[pid].update(1)
        if status == 9:
            processes[pid].join()

def main_debug(model_params):
    model_params = models_params[0]
    if model_params['adversarial']:
        model = VGGLike_Adversarial(0, mp.Queue(), model_params)
    else:
        model = VGGLike(0, mp.Queue(), model_params)
    tb_logger = pl_loggers.TensorBoardLogger(
        model_params['logdir'],
        name=model_params['logname'],
        version=''
        )
    os.makedirs(model_params['logpath']+'/checkpoints/best/', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_params['logpath']+'/checkpoints/best/',
        save_top_k=1,
        verbose=False,
        monitor='val_acc',
        mode='max',
        prefix=''
    )
    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=model_params['epochs'],
        num_sanity_val_steps=0,
        weights_summary=None,
        progress_bar_refresh_rate=0,
        track_grad_norm=1,
        gpus=1,
        # val_percent_check=0.1,
        # train_percent_check=0.5
        )
    trainer.fit(model)

    pd.DataFrame(model.history['train_loss'], columns=['train_loss']).to_csv(model_params['logpath']+'/train_loss.csv', index=False)
    pd.DataFrame(model.history['train']).to_csv(model_params['logpath']+'/train.csv', index=False)
    pd.DataFrame(model.history['val']).to_csv(model_params['logpath']+'/val.csv', index=False)

    if model_params['adversarial']:
        pd.DataFrame(model.adv_history['generator_loss']).to_csv(model_params['logpath']+'/generator_loss.csv', index=False)
        pd.DataFrame(model.adv_history['adversarial_loss']).to_csv(model_params['logpath']+'/adversarial_loss.csv', index=False)


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--params_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('-adversarial', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('-debug', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('-debug_sn', type=bool, default=False, const=True, nargs='?')
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
        logname = generate_logname(model_params)
        logpath = './' + arguments.logdir + '/' + logname
        model_params['logdir'] = arguments.logdir
        model_params['logname'] = logname
        model_params['logpath'] = logpath
        model_params['epochs'] = arguments.epochs
        model_params['adversarial'] = arguments.adversarial
        model_params['debug_sn'] = arguments.debug_sn

    if arguments.debug:
        print('DEBUUUUUUUUUUUUUUUUUUUG!!!!!!!')
        main_debug(models_params)
    else:
        print('PARALLLLLEEEEEEEEEEEEEEL!!!!!!')
        main_parallel(models_params)

    print('========= Done!!! =========')
