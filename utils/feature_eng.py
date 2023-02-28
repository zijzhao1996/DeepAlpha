import pandas as pd

# cal = calendar()
# holiday == cal.holidays(start=datetime(2015, 1, 1), end=datetime(2022, 12, 31)

# def open_close_trade(x):
#     """
#     Determine whether a price is open (9:30 AM) or close (15:00 PM)
#     """
#     return 0.5 if x.hour + x.minute / 60 == 9.5 else -0.5

# def is_holiday(x):
#     """
#     Determine a single day is holiday or not
#     """
#     return 0.5 if x.strftime('%Y-%m-%d') in holiday else -0.5

def feature_engineer(df):
    """
    Add or modify the original features in the dataframe
    """
    df.loc[:,  'time'] = pd.to_datetime(df['time'], format="%Y%m%d%H%M00000")
    df['hour_of_day'] = df['time'].apply(lambda x: (x.hour + x.minute / 60) / 23.0 - 0.5)
    df['day_of_week'] = df['time'].apply(lambda x: x.dayofweek / 6.0 - 0.5)
    df['day_of_month'] = df['time'].apply(lambda x: (x.day - 1) / 30.0 - 0.5)
    df['day_of_year'] = df['time'].apply(lambda x: (x.dayofyear - 1) / 365.0 - 0.5)
    df[ 'month_of_year']= df['time'].apply(lambda x: (x.month - 11)/11.0-0.5)
    df[ 'week_of_year']= df['time'].apply(lambda x: (x.week - 1) / 52.0 - 0.5)
    df.loc[:, 'fx461'] = df['fx461'] / 4.0 - 0.5
    df.loc[:, 'fx462'] = df['fx461'] / 113.0 - 0.5

    return df.drop([ 'time'],  axis=1)