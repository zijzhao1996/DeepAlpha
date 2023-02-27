import os
import argparse
import torch
import numpy as np
from utils.trainer import pytorch_train, lightgbm_train, xgboost_train, mtl_train
from utils.data import seed_worker, load_pytorch_train_data, load_tree_train_data
from utils.model import load_pytorch_model, count_parameters
from utils.misc import same_seed, load_config, get_logger, init_exp_setting

if __name__ == '__main__':
    # get config
    parser = argparse.ArgumentParser(description='Pytorch')
    parser.add_argument('-c',
                        '--config',
                        default='./config/LSTM/phase1/LSTM_01.yaml',
                        type=str,
                        help='config yaml file path (default: None)')
    args = parser.parse_args()
    config = load_config(args.config)

    init_exp_setting(config)
    # get logger
    logger = get_logger(
        'info',
        os.path.join(config['training']['log_save_path'], 'train_log.txt'))
    logger.info('######################################')
    logger.into('###### Welcome to use DeepAlpha ######')
    logger.info('######################################')
    logger.info('Using training mode...')

    # fixed all random seeds for reproducibility
    same_seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    # training
    logger.info('{} model is training on {}.'.format(
        config['system']['model_name'], config['system']['phase']))
    if config['system']['ispytorch']:
        ######## PyTorch model ############
        model = load_pytorch_model(config)
        logger.info(
            'Total number of parameters is {:.2f}KB ({:.2f}MB).'.format(
                count_parameters(model, "k"), count_parameters(model, "m")))
        logger.info('Constructing training model successfully.')

        train_loader, valid_loader = load_pytorch_train_data(
            config, seed_worker)
        logger.info('Loading training data successfully.')

        if config['system']['model_name'] in ['MTL', 'MTLformer']:
            mtl_train(train_loader, valid_loader, model, config, logger)
        else:
            pytorch_train(train_loader, valid_loader, model, config, logger)

    if config['system']['istree']:
        X_train, y_train, X_valid, y_valid = load_tree_train_data(config)
        logger.info('Loading training data successfully.')

        if config['system']['model_name'] == 'LightGBM':
            ##### LightGBM model ####
            model = lightgbm_train(X_train, y_train, X_valid, y_valid, config,
                                   logger)
        if config['system']['model_name'] == 'XGBoost':
            #### XGboost model ####
            # due to the limitation of current GPU memory, we downsample data only for XGboost, will improve later
            train_idx = np.random.choice(X_train.shape[0], size=3000000)
            valid_idx = np.random.choice(X_valid.shape[0], size=1000000)
            X_train, y_train = X_train[train_idx], y_train[train_idx]
            X_valid, y_valid = X_valid[valid_idx], y_valid[valid_idx]
            model = xgboost_train(X_train, y_train, X_valid, y_valid, config,
                                  logger)
        model.save_model(config['training']['model_save_path'])
    logger.info('Training ends.')
