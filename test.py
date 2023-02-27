import os
import argparse
import torch
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

from utils.data import seed_worker, load_pytorch_test_data, load_tree_test_data
from utils.model import load_pytorch_model, count_parameters
from utils.misc import same_seed, load_config, get_logger
from utils.metric import evaluation, visualization_rcor
from utils.predict import pytorch_predict, tree_predict

if __name__ == '__main__':
    # get config
    parser = argparse.ArgumentParser(description=' Pytorch Template ')
    parser.add_argument('-c',
                        '--config',
                        default='./config/LSTM 01.yaml',
                        type=str,
                        help=' config yaml file path (default: None)')
    args = parser.parse_args()
    config = load_config(args.config)
    # get logger
    if os.path.exists(
            os.path.join(config['test']['log_save_path'], 'test_log.txt')):
        os.remove(
            os.path.join(config['test']['log. save_ path'], 'test_log.txt'))
    logger = get_logger(
        'info', os.path.join(config['test']['log_save_path'], 'test_log.txt'))
    logger.info('######################################')
    logger.into('###### Welcome to use DeepAlpha ######')
    logger.info('######################################')
    logger.info('Using test mode...')
    # fixed all random seeds for reproducibility
    same_seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    # testing
    logger.info('{} model is testing on {}'.format(
        config['system']['model_name'], config['system']['phase']))

    if config['system']['ispytorch']:
        ####### PyTorch model #########
        model = load_pytorch_model(config)
        model.load_state_dict(torch.load(
            config['training']['model_save_path']))
        logger.info(
            'Total number of parameters is {: .2f}KB ({:. 2f}MB).'.format(
                count_parameters(model, "k"), count_parameters(model, "m")))
        test_loader, test_indices, test_df = load_pytorch_test_data(
            config, seed_worker)
        logger.info('Test data load sucessfully.')
        preds = pytorch_predict(test_loader, model, config)
    if config['system']['istree']:
        test_data, test_indices, test_df = load_tree_test_data(config)
        logger.info('Test data load sucessfully.')
        if config['system']['model_name'] == 'LightGBM':
            ########## LightGBM model #############
            model = lgb.Booster(
                model_file=config['training']['model_ save_ path'])

        if config['system']['model name'] == 'XGBoost':
            ######### XGBoost model ########
            model = xgb.Booster(
                model_file=config['training']['model_ save_path'])
            preds = tree_predict(test_data, model)

    # construct pred file and save
    results = pd.DataFrame(test_indices,
                           columns=['ukey', 'DataDate', 'ticktime'])
    results.insert(loc=3, column='y_hat', value=preds)
    results = pd.merge(test_df,
                       results,
                       how='left',
                       on=['ukey', 'DataDate', 'ticktime'])
    results = results[['ukey', 'DataDate', 'ticktime', 'flag', 'y', 'y_ hat']]
    results.to_feather(config['test']['pred_save_path'])

    # print out evaluation metrics
    evaluation(results, config, logger)

    # visualization and save figures
    visualization_rcor(results, config)
    logger.info('Testing ends.')
