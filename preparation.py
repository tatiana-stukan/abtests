from functools import reduce

import pandas as pd


def aggregate(column: str, metric: str):
    def do(df: pd.DataFrame, experiment) -> pd.DataFrame:
        return df.groupby(['user_id', experiment], as_index=False).agg(**{
            metric: (column, 'sum')
        })

    return do


def aggregate_user_retention(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    first_day = df.groupby(['user_id'])['date'].min()
    df['first_day'] = df['user_id'].map(first_day)
    df['next_day'] = (df['date'] == df['first_day'] + pd.Timedelta(days=1)).astype(int)

    df = df.groupby(['user_id', experiment], as_index=False).agg(
        user_retention=('next_day', 'sum')
    )

    df['user_retention'] = (df['user_retention'] > 0).astype(int)

    return df


metrics_agg_map = {
    'arpu': aggregate('price_usd', 'arpu'),
    'messages': aggregate('messages_count', 'messages'),
    'user_retention': aggregate_user_retention
}


def exclude_experiment_hopping_users(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    df_count_before_filter = df.shape[0]
    df = df[
        df.groupby('user_id')[experiment].transform(lambda x: x.min() == x.max())
    ]

    if df_count_before_filter != df.shape[0]:
        print(f'Warning! Users change their A/B groups')

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
    data = metrics_agg_map[metric](df, experiment)

    a = data[data[experiment] == 0][metric]
    b = data[data[experiment] == 1][metric]

    return a, b
