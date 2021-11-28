import math

import numpy as np
import pandas as pd
import xlearn as xl
from sklearn.preprocessing import OrdinalEncoder
from xlearn import XLearn


# noinspection PyUnresolvedReferences
def feature_engineering(df: pd.DataFrame, zone_id: dict) -> pd.DataFrame:
    """Perform FE on the given dataframe.

    We add hour, weekday and daytime features for each event.
    All categorical features we encode with pandas.get_dummies.

    Args:
        df: dataframe to perform feature engineering at
        zone_id: dictionary to map zone_id column with

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
    enc = OrdinalEncoder()
    table.oaid_hash = enc.fit_transform(table[["oaid_hash"]]).astype(int)
    table["hour"] = table.index.hour
    table["weekday"] = table.index.weekday
    table["weekend"] = (table.weekday > 4).astype(int)
    table["daytime"] = table.hour.progress_apply(get_daytime)
    table.zone_id = table.zone_id.map(zone_id).fillna(np.mean(list(zone_id.values())))
    return table


def convert_to_ffm(path, df, type, target, numerics, categories, features, encoder):
    """https://github.com/wngaw/blog/blob/master/xlearn_example/src/utils.py"""
    for x in numerics:
        if x not in encoder["catdict"]:
            encoder["catdict"][x] = 0
    for x in categories:
        if x not in encoder["catdict"]:
            encoder["catdict"][x] = 1

    nrows = df.shape[0]
    with open(path + str(type) + "_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow[target]))  # Set Target Variable here

            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(encoder["catdict"].keys()):
                if encoder["catdict"][x] == 0:
                    # Not adding numerical values that are nan
                    if math.isnan(datarow[x]) is not True:
                        datastring = (
                            datastring
                            + " "
                            + str(i)
                            + ":"
                            + str(i)
                            + ":"
                            + str(datarow[x])
                        )
                else:
                    # For a new field appearing in a training example
                    if x not in encoder["catcodes"]:
                        encoder["catcodes"][x] = {}
                        encoder["currentcode"] += 1
                        encoder["catcodes"][x][datarow[x]] = encoder[
                            "currentcode"
                        ]  # encoding the feature

                    # For already encoded fields
                    elif datarow[x] not in encoder["catcodes"][x]:
                        encoder["currentcode"] += 1
                        encoder["catcodes"][x][datarow[x]] = encoder[
                            "currentcode"
                        ]  # encoding the feature

                    code = encoder["catcodes"][x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            datastring += "\n"
            text_file.write(datastring)
    print("File written...")
    return encoder


# noinspection NonAsciiCharacters
def create_model(k: int, λ: float = 0.001) -> XLearn:
    """Create FFM model with the given parameters and fit it on the given data.

    Args:
        k: dimension parameter
        λ: lambda parameter

    Returns:
        trained FFM model on the given data
    """
    ffm_model = xl.create_ffm()
    ffm_model.setTrain("../data/train_ffm.txt")
    ffm_model.setValidate("../data/val_ffm.txt")
    ffm_model.fit(
        {"task": "binary", "lr": 1.0, "lambda": λ, "k": k, "metric": "acc"},
        model_path="./model.out",
    )
    return ffm_model
