from functools import reduce

import pandas as pd

simple_metrics_column_map = {
    'arpu': 'price_usd',
    'messages': 'messages_count',
}


def exclude_experiment_hopping_users(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    df = df[
        df.groupby('user_id')[experiment].transform(lambda x: x.min() == x.max())
    ]

    return df[df[experiment] >= 0]


def prepare_for_experiment(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    df[experiment] = df[experiment].fillna(-1).map({
        -1: -1,
        False: 0,
        True: 1
    })

    preparations = [
        exclude_experiment_hopping_users
    ]
    df = reduce(lambda acc, preparation: preparation(acc, experiment), preparations, df)

    return df


def get_aggregated_a_b_groups(df: pd.DataFrame, experiment: str, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.groupby(['user_id', experiment], as_index=False).agg(**{
        metric: (simple_metrics_column_map[metric], 'sum')
    })

    a = data[data[experiment] == 0][metric]
    b = data[data[experiment] == 1][metric]

    return a, b
