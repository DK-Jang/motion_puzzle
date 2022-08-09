from etc.utils import set_seed, ensure_dirs, get_config
from data_loader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
import shutil
import argparse
import os
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)


def initialize_path(args, config, save=True):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    config['tb_dir'] = os.path.join(config['main_dir'], "log")
    config['info_dir'] = os.path.join(config['main_dir'], "info")
    config['output_dir'] = os.path.join(config['main_dir'], "output")
    ensure_dirs([config['main_dir'], config['model_dir'], config['tb_dir'],
                 config['info_dir'], config['output_dir']])
    if save:
        shutil.copy(args.config, os.path.join(config['info_dir'], 'config.yaml'))


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help='Path to the config file.')
args = parser.parse_args()

""" initialize """
config = get_config(args.config)
initialize_path(args, config)

# Set random seed for reproducibility
print("Random Seed: ", config['manualSeed'])
set_seed(config['manualSeed'])

""" Dataloader """
train_src_loader = get_dataloader('train', config)
train_tar_loader = get_dataloader('train', config)
loader = {'train_src': train_src_loader, 'train_tar': train_tar_loader}

""" Summary Writer """
train_writer = SummaryWriter(os.path.join(config['tb_dir'], 'train'))

# Trainer
trainer = Trainer(config)
tr_info = open(os.path.join(config['info_dir'], "info-network"), "w")
print(trainer.gen, file=tr_info)
tr_info.close()
trainer.train(loader, train_writer)