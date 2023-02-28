from model import *
import yaml
import os
import torch
import numpy as np
import logging
import sys
import random
import shutil

sys.path.append('..')


def load_config(config_name):
    """
    Load config file.
    """
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def init_exp_setting(config):
    """
    Create new subfolders for config, if already exist, delete it first.
    """
    for folder in ['analysis', 'ckpts', 'logs', 'preds', 'tsboards']:
        subfolder = os.path.join('./results/{}/ '.format(folder),
                                 config['system']['model_name'],
                                 config['system']['phase'],
                                 config['system']['experimental_id'])
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        os.makedirs(subfolder)


def get_logger(level, log_file=None):
    """
    Initiate logger file and schema.
    """
    head = '[%(asctime)-15s] [%(levelnames)] %(messages)s'
    if level == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif level == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file != None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger


def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)