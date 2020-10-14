import os
import pickle
import pandas as pd
import numpy as np

from .data import data_path, resource

__all__ = ['load']


# Keep fixed for reproducibility.
np.random.seed(1)
cache_weekly = data_path('japan', 'weekly', 'experiment.pickle')


def load():
    _fetch()

    # If path does not exists, generate DataFrames.
    if not os.path.exists(cache_weekly):
        _parse()

    with open(cache_weekly, 'rb') as f:
        return pickle.load(f)


def _fetch():
    # All data.
    resource(
        target=data_path('japan', '80s.csv'),
        url='https://www.dropbox.com/s/kokgr6ekb0gh7wq/japan_80s.csv?dl=1')


def _parse():
    df = pd.read_csv(data_path('japan', '80s.csv'))

    # Extract 1980 and 1981 data.
    df = df[(df['date'].str[:4] == '1980') | (df['date'].str[:4] == '1981')]

    # Rescale TAVG.
    df['TAVG'] = df['TAVG'] / 10

    # Convert dates to Datetime.
    df['date'] = pd.DatetimeIndex(df['date'])

    # Create weekly groups.
    n = 7
    df['week'] = (df['date'] - df.iloc[0]['date']).dt.days // n
    df['day'] = (df['date'] - df.iloc[0]['date']).dt.days % n

    observations = ['TMAX', 'TMIN', 'TAVG']
    test_idx = [[] for _ in observations]
    for week, week_df in df.groupby('week'):
        # Set TAVG for middle three days to missing.
        mid_df = week_df[week_df['day'].isin([1, 2, 3, 4, 5])]
        test_idx[2] += list(mid_df.index)

        # Set 25% of TMAX and TMIN values to missing.
        m = len(week_df) // 4
        test_idx[0] += list(np.random.choice(list(week_df.index), m,
                                             replace=False))
        test_idx[1] += list(np.random.choice(list(week_df.index), m,
                                             replace=False))

    # Train indices are all those that aren't in test indices.
    train_idx = [list(set(df.index.tolist()) - set(idx)) for idx in test_idx]

    train = df.copy()
    test = df.copy()

    for obs, train_idx_, test_idx_ in zip(observations, train_idx, test_idx):
        train[obs][test_idx_] = np.nan
        test[obs][train_idx_] = np.nan

    # Set all other nans in test df.
    for obs in ['PRCP', 'SNWD']:
        test[obs][:] = np.nan

    train_1980 = train[train['date'].dt.year == 1980]
    train_1981 = train[train['date'].dt.year == 1981]
    test_1980 = test[test['date'].dt.year == 1980]
    test_1981 = test[test['date'].dt.year == 1981]

    # Train on all data from 1980.
    train = df[df['date'].dt.year == 1980]

    data = {'all': df}
    names = ['train', 'train_1980', 'train_1981', 'test_1980', 'test_1981']
    dfs = [train, train_1980, train_1981, test_1980, test_1981]
    # Extract stations per group.
    for name, df in zip(names, dfs):
        per_week = []
        for week, week_df in df.groupby('week'):
            per_week.append(week_df.copy())

        data[name] = per_week

    # Extract dates.
    data['dates_1980'] = [x.iloc[0].week for x in data['train_1980']]
    data['dates_1981'] = [x.iloc[0].week for x in data['train_1981']]

    # Save experiment data.
    if not os.path.exists(os.path.dirname(cache_weekly)):
        os.makedirs(os.path.dirname(cache_weekly))

    with open(cache_weekly, 'wb') as f:
        pickle.dump(data, f)
