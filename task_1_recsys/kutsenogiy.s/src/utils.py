import itertools
import typing as tp
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.base_config import Config


def date_to_week_day(date: str) -> int:
    return datetime.strptime(date.split()[0], '%Y-%m-%d').weekday()


def feature_to_mean(
        df: pd.DataFrame,
        features: tp.List,
        feature_name: str,
) -> pd.DataFrame:
    features2uniq = []
    for feature in features:
        features2uniq.append(np.unique(df[feature].values))

    transform_dict = defaultdict(float)
    for values in itertools.product(*features2uniq):
        tmp_df = deepcopy(df)
        for i, feature in enumerate(features):
            tmp_df = tmp_df[tmp_df[feature] == values[i]]
        transform_dict[values] = np.sum(tmp_df.clicks) / len(tmp_df)

    df[feature_name] = df.apply(lambda x: transform_dict[tuple(x[name] for name in features)], axis=1).values
    return df


def prepare_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    for new_feature in tqdm(config.features_to_generate, 'Features in process'):
        df = feature_to_mean(df, **new_feature)
    return df


def prepare_dates_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['weekday'] = df.date_time.apply(date_to_week_day)
    df['hours'] = df['date_time'].apply(lambda x: x.split()[-1].split(':')[0])
    return df


def filter_by_date(df: pd.DataFrame, date: str) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    filter_part = df.date_time.apply(lambda x: x.split(" ")[0] == date)
    return df[~filter_part], df[filter_part]


def prepare_dateframe(config: Config) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.data_path).sample(config.size, random_state=config.random_state)
    df = prepare_dates_feature(df)
    df = prepare_features(df, config)
    train_df, test_df = filter_by_date(df, config.test_date)
    return train_df[config.features_to_train], test_df[config.features_to_train]
