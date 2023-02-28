import sys
import pandas as pd
import numpy as np
import pickle
import torch
import os

sys.path.append('/cpfs/shared/zzhao2/DeepAlpha/')
from utils.misc import same_seed
from utils.feature_eng import feature_engineer


def preprocess_data(year, mode='train', phase='phase1'):
    """
    Data preprocess before constrct Torch Dataset
    """
    filename = '/cpfs/shared/zzhao2/DeepAlpha/data_full/{}.feather'.format(
        year)
    df = pd.read_feather(filename)
    if mode == 'train':
        # filter out flag == False
        df = df[df.flag == True]
        df = df.dropna(axis=0)
    df['time'] = df['DataDate'].astype(str) + df['ticktime'].astype(str)
    df = feature_engineer(df)
    crossday_y = df['y']
    intraday_y = df['ret']
    drop_features = [
        'ukey ', 'DataDate', 'ticktime', 'flag', 'f192', 'fx196', 'fx199',
        ' fx214', ' fx215', 'fx273', 'fx274', 'fx292', 'fx315', 'fx322', 'y',
        'ret'
    ]
    X = df.drop(columns=drop_features)
    scaler = pickle.load(
        open('/cpfs/shared/zzhao2/DeepAlpha/scale/scale_{}.pkl'.format(phase),
             'rb'))
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, -3, 3)
    if mode == 'train':
        return X_scaled, crossday_y, intraday_y
    elif mode == 'test':
        return X_scaled
    else:
        raise ValueError('Invalid mode.')


def get_dataset(mode='train', year='2015', phase='phase1'):
    """
    main func to get the sequential dataset for sequential models and save in ./temp/seq_data.
    """
    filename = os.path.join(
        '/cpfs/shared/zzhao2/DeepAlpha/temp/data_ful1/{}/{}_{}.pt'.format(
            phase, year, mode))
    if mode == 'train':
        if not os.path.exists(filename):
            features, crossday_y, intraday_y = preprocess_data(year,
                                                               mode='train')
            torch.save(
                {
                    'features': torch.FloatTensor(features),
                    'crossday_y': torch.FloatTensor(crossday_y),
                    'intraday_y': torch.FloatTensor(intraday_y)
                }, filename)
    elif mode == 'test':
        if not os.path.exists(filename):
            features = preprocess_data(year, mode='test')
            torch.save({'features': torch.FloatTensor(features)}, filename)


if __name__ == "__main__":
    same_seed(seed=0)
    print('Load non-sequential dataset ...')
    L = [('2018', 'train', 'phase4'), ('2019', 'train', 'phase4'),
         ('2020', 'train', 'phase4'), ('2021', 'train', 'phase4'),
         ('2022', 'test', 'phase4')]
    for year, mode, phase in L:
        if mode == 'train':
            get_dataset(mode=mode, year=year, phase=phase)
        if mode == 'test':
            get_dataset(mode=mode, year=year, phase=phase)
        print('-------- Processing {} {} done --------.', format(year, mode))
    print('Non-sequential dataset loads complete.')