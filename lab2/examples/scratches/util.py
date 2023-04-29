import numpy as np
import pandas as pd

pd.options.display.max_rows = 5000
pd.options.display.max_columns = 20
import json


def read_file(name: str) -> pd.DataFrame:
    with open(f"../../data/3analize_{name}.json", "r") as f:
        dt = json.loads(f.read())
    return pd.DataFrame(dt)


def get_alpha(df: pd.DataFrame):
    return df["alpha"].unique()


def get_df_with_count_alpha_and_run(df: pd.DataFrame):
    return df.groupby(['count', 'alpha'])['run'].apply(lambda x: x)


def get_nan_with_count_alpha(df: pd.DataFrame):
    return df.groupby(['count', 'alpha'])['result_err'].apply(lambda x: x.isnan().sum())


def get_mean_with_count_alpha_and_run(df: pd.DataFrame):
    return df.groupby(['count', 'alpha'])['run'].apply(lambda x: x.mean())


# df = read_file("adagrad")
# print(df.head())
# df_na = df.dropna()
# print(df_na)
# df_na["math_iters"].filter(lambda x: x<10_000)
# arrays = [
#     df["alpha"].unique(),
#     df["count"].unique(),
# ]
# index = pd.MultiIndex.from_product(arrays)

