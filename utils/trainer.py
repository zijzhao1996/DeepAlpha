import torch
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import lightgbm as lgb
import xgboost as xgb
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from .loss import custom_loss
from .metric import pearsonr
from .predict import tree_predict


def pytorch_train(train_loader, valid_loader, model, config, logger):
    """
    train function for deep learning models built via PyTorch
    """
    # define loss
    if config['training']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss'] == 'mae':
        criterion = nn.L1Loss()
    elif config['training']['loss'] == 'custom':
        criterion = custom_loss
    else:
        raise Exception("unknown loss {}".format(config['training']['loss']))

    # define optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'])
    if config['training']['scheduler'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.9,
                                                           verbose=True)
    elif config['training']['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_O=2,
            T_mult=2,
            eta_min=config['training']['learning_rate'])
    else:
        raise Exception("unknown schedular {}".format(
            config['training']['scheduler']))

    # writer of tensoboard
    writer = SummaryWriter(config['training']['tsboard_save_path'])

    # training
    n_epochs, best_loss, step, early_stop_count = config['training'][
        'n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        ic_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(config['system']['device']), y.to(
                config['system']['device'])
            # special training for TabNet
            if config['system']['model_name']['TabNet']:
                priors = torch.ones(x.shape[0], config['model']['inp_dim']).to(
                    config['system']['device'])
                pred = model(x, priors)
                pred = pred[0].squeeze(1)
            else:
                pred = model(x)

            # special training for MLPMoE
            if config['system']['model_name'] == 'MLPMoE':
                pred, aux_loss = pred
                loss = criterion(pred, y)
                loss = loss + aux_loss
            else:
                loss = criterion(pred, y)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            except Exception as e:
                pass
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
            ic = pearsonr(pred.detach(), y.detach())
            ic_record.append(ic)
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss ': loss.detach().item()})
        scheduler.step()
        # update tensorboard
        mean_train_loss = sum(loss_record / len(loss_record))
        mean_train_ic = sum(ic_record / len(ic_record))
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('IC/train', mean_train_ic, step)

        # validation
        model.eval()
        loss_record = []
        ic_record = []
        for x, y in valid_loader:
            x, y = x.to(config['system']['device']), y.to(
                config['system']['device'])
            with torch.no_grad():
                if config['system']['model_name'] == 'TabNet':
                    priors = torch.ones(x.shape[0],
                                        config['model']['inp_dim']).to(
                                            config['system']['device'])
                    pred = model(x, priors)
                    pred = pred[0].squeeze(1)
                else:
                    pred = model(x)
                # special training for MLPMoE
                if config['system']['model_name'] == 'MLPMoE':
                    pred, aux_loss = pred
                    loss = criterion(pred, y)
                    loss = loss + aux_loss
                else:
                    loss = criterion(pred, y)
                ic = pearsonr(pred.detach(), y.detach())
        loss_record.append(loss.item())
        ic_record.append(ic)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        mean_valid_ic = sum(ic_record) / len(ic_record)
        logger.info(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.6f},Train IC:{mean_train_ic:.6f}, Valid loss: {mean_valid_loss: .6f}, valid IC: {mean_valid_ic: .6f}'
        )
        # update on tensorboard
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('IC/valid', mean_valid_ic, step)

        # model save
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),
                       config['training']['model_save_path'])
            logger.info('Saving model with loss {:.6f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        # early stop
        if early_stop_count >= config['training']['early_stop']:
            logger.info(
                '\nModel is not improving, so we halt the training session.')
            return


def mtl_train(train_loader, valid_loader, model, config, logger):
    """
    train function for deep learning models built via PyTorch
    """
    # define loss
    if config['training']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss'] == 'mae':
        criterion = nn.L1Loss()
    elif config['training']['loss'] == 'huber':
        criterion = nn.HuberLoss(reduction='mean', delta=0.1)
    elif config['training']['loss'] == 'custom':
        criterion = custom_loss
    else:
        raise Exception("unknown loss {}".format(config['training']['loss']))

    # define optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'])
    if config['training']['scheduler'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.9,
                                                           verbose=True)
    elif config['training']['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_O=2,
            T_mult=2,
            eta_min=config['training']['learning_rate'])
    else:
        raise Exception("unknown schedular {}".format(
            config['training']['scheduler']))

    # writer of tensoboard
    writer = SummaryWriter(config['training']['tsboard_save_path'])

    # training
    n_epochs, best_loss, step, early_stop_count = config['training'][
        'n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        ic_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, cross_y, intra_y in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            x, cross_y, intra_y = x.to(config['system']['device']), cross_y.to(
                config['system']['device']), intra_y.to(
                    config['system']['device'])
            cross_pred, intra_pred = model(x)
            loss = criterion(
                cross_pred,
                cross_y) + config['training']['loss_lambda'] * criterion(
                    intra_pred, intra_y)
            loss.backward()
        try:
            torch.nn.utils.clip_grad_value_(model.parameters, 3.0)
        except Exception as e:
            pass
        optimizer.step()
        step += 1
        loss_record.append(loss.detach().item())
        ic = pearsonr(cross_pred.detach(), cross_y.detach())
        ic_record.append(ic)
        train_pbar.set_description(f'Epoch[{epoch+1}/{n_epochs}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})
        scheduler.step()

        # update on tensorboard
        mean_train_loss = sum(loss_record) / len(loss_record)
        mean_train_ic = sum(ic_record) / len(ic_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('IC/train', mean_train_ic, step)

        # validation
        model.eval()
        loss_record = []
        ic_record = []
        for x, cross_y, intra_y in valid_loader:
            x, cross_y, intra_y = x.to(config['system']['device']), cross_y.to(
                config['system']['device']), intra_y.to(
                    config['system']['device'])
            with torch.no_grad():
                cross_pred, intra_pred = model(x)
                loss = criterion(
                    cross_pred,
                    cross_y) + config['training']['loss_lambda'] * criterion(
                        intra_pred, intra_y)
            ic = pearsonr(cross_pred.detach(), cross_y.detach())
            loss_record.append(loss.item())
            ic_record.append(ic)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        mean_valid_ic = sum(ic_record) / len(ic_record)
        logger.info(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.6f},Train IC:{mean_train_ic:.6f}, Valid loss: {mean_valid_loss: .6f}, valid IC: {mean_valid_ic: .6f}'
        )
        # update on tensorboard
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('IC/valid', mean_valid_ic, step)

        # model save
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),
                       config['training']['model_save_path'])
            logger.info('Saving model with loss {:.6f}..'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['training']['early_stop']:
            logger.info(
                '\nModel is not improving, so we halt the training session.')
            return


def pytorch_train_toy(train_loader, valid_loader, model, config, logger):
    """
    train function for deep learning models built via PyTorch
    """
    # define loss
    if config['training']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss'] == 'mae':
        criterion = nn.L1Loss()
    elif config['training']['loss'] == 'custom':
        criterion = custom_loss
    else:
        raise Exception("unknown loss {}".format(config['training']['loss']))

    # define optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'])
    if config['training']['scheduler'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.9,
                                                           verbose=True)
    elif config['training']['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_O=2,
            T_mult=2,
            eta_min=config['training']['learning_rate'])
    else:
        raise Exception("unknown schedular {}".format(
            config['training']['scheduler']))

    # writer of tensoboard
    writer = SummaryWriter(config['training']['tsboard save_path'])

    # training
    n_epochs, best_loss, step, early_stop_count = config['training'][
        'n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        ic_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(config['system']['device']), y.to(
                config['system']['device'])
            if config['system']['model_name'] == 'TabNet ':
                priors = torch.ones(x.shape[0], config['model']['inp_dim']).to(
                    config['system']['device'])
                pred = model(x, priors)
                pred = pred[0].squeeze(1)
            else:
                pred = model(x)
            # specialtraining for MLPMoE
            if config['system']['model_name'] == 'MLPMoE':
                pred, aux_loss = pred
                loss = criterion(pred, y)
                loss = loss + aux_loss
            else:
                loss = criterion(pred, y)
                loss.backward()
            try:
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            except Exception as e:
                pass
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
            ic = pearsonr(pred.detach(), y.squeeze().detach())
            ic_record.append(ic)
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        scheduler.step()

        # update on tensorboard
        mean_train_loss = sum(loss_record) / len(loss_record)
        mean_train_ic = sum(ic_record) / len(ic_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('IC/train', mean_train_ic, step)

        # validation
        model.eval()
        loss_record = []
        ic_record = []
        for x, y in valid_loader:
            x, y = x.to(config['system']['device']), y.to(
                config['system']['device'])
            with torch.no_grad():
                if config['system']['model_name'] == 'TabNet':
                    priors = torch.ones(x.shape[0],
                                        config['model']['inp_dim']).to(
                                            config['system']['device'])
                    pred = model(x, priors)
                    pred = pred[0].squeeze(1)
                else:
                    pred = model(x)
                # special training for MLPMoE
                if config['system']['model_name'] == 'MLPMoE':
                    pred, aux_loss = pred
                    loss = criterion(pred, y)
                    loss = loss + aux_loss
                else:
                    loss = criterion(pred, y)
                ic = pearsonr(pred.detach(), y.squeeze().detach())
            loss_record.append(loss.item())
            ic_record.append(ic)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        mean_valid_ic = sum(ic_record) / len(ic_record)
        logger.info(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.6f},Train IC:{mean_train_ic:.6f}, Valid loss: {mean_valid_loss: .6f}, valid IC: {mean_valid_ic: .6f}'
        )
        # update on tensorboard
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('IC/valid', mean_valid_ic, step)

        # model save
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),
                       config['training']['model_save_path'])
            logger.info('Saving model with loss {:.6f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['training']['early_stop']:
            logger.info(
                '\nModel is not improving, so we halt the training session.')
            return


def mtl_train_toy(train_loader, valid_loader, model, config, logger):
    """
    train function for deep learning models built via PyTorch
    """
    # define loss
    if config['training']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss'] == 'mae':
        criterion = nn.L1Loss()
    elif config['training']['loss'] == 'huber':
        criterion = nn.HuberLoss(reduction='mean', delta=0.1)
    elif config['training']['loss'] == 'custom':
        criterion = custom_loss
    else:
        raise Exception("unknown loss {}".format(config['training']['loss']))

    # define optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training ']['weight_decay'])
    if config['training']['scheduler'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.9,
                                                           verbose=True)
    elif config['training']['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_O=2,
            T_mult=2,
            eta_min=config['training']['learning_rate'])
    else:
        raise Exception("unknown schedular {}".format(
            config['training']['scheduler']))

    # writer of tensoboard
    writer = SummaryWriter(config['training']['tsboard_save_path'])

    # training
    n_epochs, best_loss, step, early_stop_count = config['training'][
        'n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        ic_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, cross_y, intra_y in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            x, cross_y, intra_y = x.to(config['system']['device']), cross_y.to(
                config['system']['device']), intra_y.to(
                    config['system']['device'])
            cross_pred, intra_pred = model(x)
            loss = criterion(cross_pred, cross_y) + criterion(
                intra_pred, intra_y)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            except Exception as e:
                pass
            optimizer.step
            step += 1
            loss_record.append(loss.detach().item())
            ic = pearsonr(cross_pred.detach(), cross_y.squeeze().detach())
            ic_record.append(ic)
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        scheduler.step()

        # update on tensorboard
        mean_train_loss = sum(loss_record) / len(loss_record)
        mean_train_ic = sum(ic_record) / len(ic_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('IC/train', mean_train_ic, step)

        # validation
        model.eval()
        loss_record = []
        ic_record = []
        for x, cross_y, intra_y in valid_loader:
            x, cross_y, intra_y = x.to(config['system']['device']), cross_y.to
            (config['system']['device']), intra_y.to(
                config['system']['device'])
            with torch.no_grad():
                cross_pred, intra_pred = model(x)
                loss = criterion(cross_pred, cross_y) + criterion(
                    intra_pred, intra_y)
                ic = pearsonr(cross_pred.detach(), cross_y.squeeze().detach())
                loss_record.append(loss.item())
                ic_record.append(ic)
                mean_valid_loss = sum(loss_record) / len(loss_record)
                mean_valid_ic = sum(ic_record) / len(ic_record)
            logger.info(
                f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.6f},Train IC:{mean_train_ic:.6f}, Valid loss: {mean_valid_loss: .6f}, valid IC: {mean_valid_ic: .6f}'
            )
        # update on tensorboard
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('IC/valid', mean_valid_ic, step)

        # model save
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),
                       config['training']['model_save_path'])
            logger.info('Saving model with loss {:.6f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['training']['early_stop']:
            logger.info(
                '\nModel is not improving, so we halt the training session.')
        return


def lightgbm_train(X_train, y_train, X_valid, y_valid, config, logger):
    """
    Train function of lightGBM
    """
    params = {
        'poosting_type': config['model']['boosting_type'],
        'loss': config['model']['loss'],
        'Lambda__l1': config['model']['lambda_l1'],
        '1ambda_12': config['model']['1ambda_12'],
        'max_depth': config['nmode1']['max_depth'],
        'num leaves': config[' model']['num_1eaves '],
        'num_threads': config['model']['num_threads '],
        'metric': config['model']['metric'],
        'num_leaves': config['model']['num_leaves'],
        'learning_rate': config['model']['learning_rate'],
        'feature_fraction': config['model']['feature_fraction'],
        'bagging_fraction': config['model']['bagging_fraction'],
        'bagging_freq': config['model']['bagging_freq'],
        'verbose': config['model']['verbose']
    }
    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=config['training']['num_boost_round'],
        valid_sets=valid_data,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=config['training']['stopping_rounds'])
        ])
    # print loss and ic on train/valid set
    train_preds = tree_predict(X_train, gbm)
    valid_preds = tree_predict(X_valid, gbm)
    train_loss = mean_squared_error(train_preds, y_train)
    valid_loss = mean_squared_error(valid_preds, y_valid)
    train_ic = pearsonr(train_preds, y_train, type='numpy')
    valid_ic = pearsonr(valid_preds, y_valid, type='numpy')
    logger.info(
        f'Train loss:{train_loss:.6f},Train IC:{train_ic:.6f}, Valid loss:{valid_loss:.6f}, Valid IC: {valid_ic:.6f}'
    )
    return gbm


def xgboost_train(X_train, y_train, X_valid, y_valid, config, logger):
    """
    Train function of XGBoost mdel
    """
    params = {
        'booster': config['model']['booster'],
        'objective': config['model']['objective'],
        'gamma': config['model']['gamma'],
        'max_depth': config['model']['max_depth'],
        'lambda': config['mode1']['lambda'],
        'subsample': config['model']['subsample'],
        'colsample_bytree': config['model']['colsample_bytree'],
        'min_child_weight': config['model']['min_child_weight'],
        'verbosity': config['model']['verbosity'],
        'eta': config['model']['eta'],
        'nthread': config['model']['nthread'],
        'tree_method': config['model']['tree_method'],
        'gpu_id': config['model']['gpu_id']
    }
    train_data = xgb.DMatrix(X_train, y_train)
    valid_data = xgb.DMatrix(X_valid, y_valid)
    xgbmodel = xgb.train(params,
                         train_data,
                         evals=[(train_data, 'Train'), (valid_data, 'Valid')],
                         num_boost_round=config['training']['num_boost_round'],
                         callbacks=[
                             xgb.callback.EarlyStopping(
                                 rounds=config['training']['stopping_rounds'])
                         ])

    # print loss/ic on train/valid set
    train_preds = tree_predict(train_data, xgbmodel)
    valid_preds = tree_predict(valid_data, xgbmodel)
    train_loss = mean_squared_error(train_preds, y_train)
    valid_loss = mean_squared_error(valid_preds, y_valid)
    train_ic = pearsonr(train_preds, y_train, type='numpy')
    valid_ic = pearsonr(valid_preds, y_valid, type='numpy')
    logger.info(
        f'TrainLoss:{train_loss:.6f}, Train IC: {train_ic:.6f}, Valid loss: {valid_loss:.6f}, Valid IC: {valid_ic:.6f}'
    )
    return xgbmodel