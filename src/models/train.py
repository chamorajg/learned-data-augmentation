import os
import sys
import yaml
import argparse
import numpy as np
sys.path.append('../../')
from models import *
import vae_experiments
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='../configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = vae_experiments[config['model_params']['experiment']](**config['exp_params'])
if 'checkpoint_callback' in config:
    checkpoint_callback = ModelCheckpoint(monitor=config['checkpoint_callback']['monitor'],
                                        mode='max')
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                    logger=tt_logger,
                    log_save_interval=100,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    early_stop_callback = False,
                    gradient_clip_val=config['exp_params']['gradient_clip'],
                    checkpoint_callback=checkpoint_callback,
                    **config['trainer_params'])
else:
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                    logger=tt_logger,
                    log_save_interval=100,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    early_stop_callback = False,
                    gradient_clip_val=config['exp_params']['gradient_clip'],
                    **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)