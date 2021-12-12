from typing import Tuple, Union

import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def get_pi(e, d):
    """Calculate pi_*."""
    return norm.sf((0 - e) / d, e, d)


def get_cips(p0, p1, data, la=10):
    """Calculate cips."""
    return (np.minimum(np.nan_to_num(p1/p0), la) * data.clicks.values).sum() / data.shape[0]


# noinspection PyUnresolvedReferences
def feature_engineering(df: pd.DataFrame, oh: OneHotEncoder, t: TargetEncoder) -> pd.DataFrame:
    """Perform FE on the given dataframe.

    We add hour, weekday and daytime features for each event.
    All categorical features we encode with pandas.get_dummies.

    Args:
        df: dataframe to perform feature engineering at
        ohe: OneHotEncoder for the data
        t: TargetEncoder

    Returns:
        prepared dataframe with features
    """

    def get_daytime(x: int) -> str:
        """Get daytime from the given hour."""
        if (x > 4) and (x <= 8):
            return "Early Morning"
        elif (x > 8) and (x <= 12):
            return "Morning"
        elif (x > 12) and (x <= 16):
            return "Noon"
        elif (x > 16) and (x <= 20):
            return "Eve"
        elif (x > 20) and (x <= 24):
            return "Night"
        elif x <= 4:
            return "Late Night"

    table = df.copy(deep=True)
    table["hour"] = table.index.hour
    table["weekday"] = table.index.weekday
    table[['zone_id','banner_id']] = t.transform(table[['zone_id','banner_id']])
    return pd.concat([table[['hour','zone_id','banner_id','clicks', 'weekday']], oh.transform(table[['os_id','country_id']])], axis=1)


def split_data_labels(
    df: pd.DataFrame, target: str = "clicks"
) -> Tuple[np.ndarray, ...]:
    """Split the given dataframe to data and labels.

    Args:
        df: dataframe to split
        target: labels column name
    Returns:
        tuple of ndarrays: data and labels.
    """
    return df.drop([target], axis=1).values, df[target].values


def create_model(df: pd.DataFrame, c: float) -> LogisticRegression:
    """Create LR-model and fit it on the given data.

    Args:
        df: data to train model on
        c: C parameter for LR model
        penalty: penalty type for LR model

    Returns:
        trained LR model on the given data
    """
    x, y = split_data_labels(df)
    return LogisticRegression(
        random_state=42, C=c, tol=0.01, max_iter=200
    ).fit(x, y)


def score(y_true: np.ndarray, y_pred: Union[np.ndarray, list]) -> float:
    """Calculate the score (in this task it's log_loss).

    Args:
        y_true: true labels
        y_pred: predicted

    Returns:
        the results score
    """
    return log_loss(y_true, y_pred)


def test_model(df: pd.DataFrame, model) -> float:
    """Test model on the given test dataframe.

        df: data to test model on
        model: the model to test

    Returns:
        score of the model on the the test data
    """
    test_data, ground_truth = split_data_labels(df)
    return score(ground_truth, model.predict_proba(test_data))
