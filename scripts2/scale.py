import sys
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.append('/cpfs/shared/zzhao2/DeepAlpha/')
from utils.feature_eng import feature_engineer


def load_scale(df, phase):
    """
    main func to load scale in ./scale folder
    """
    df = df.fillna(0)
    # concat DataDate & ticktime
    df['time'] = df['DataDate'].astype(str) + df['ticktime'].astype(str)
    df = feature_engineer(df)
    drop_cols = [
        'ukey', 'DataDate', 'ticktime', 'f192', 'fx196', 'fx199', 'fx214',
        'fx215', 'fx273', 'f274', 'f292', 'fx315 ', 'fx322', 'flag', 'y', 'ret'
    ]
    df = df.drop(drop_cols, axis=1)
    scale = RobustScaler().fit(df)
    pickle.dump(
        scale,
        open('/cpfs/shared/zzhao2/DeepAlpha/scale/scale_{}.pkl'.format(phase),
             'wb'))


if __name__ == '__main__':
    # only use training part to compute the mean and std for the scale
    YEAR_MAP = {'phase4': [2018, 2019, 2020]}
    df_list = []
    for year in YEAR_MAP['phase4']:
        df_list.append(
            pd.read_feather(
                '/cpfs/shared/zzhao2/DeepAlpha/data_full/{}.feather'.format(
                    year)))
    result = pd.concat(df_list, axis=0)
    result = result.reset_index(0, drop=True)
    print(result.shape)
    load_scale(result, 'phase4')