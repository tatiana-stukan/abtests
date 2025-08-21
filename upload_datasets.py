from datetime import datetime, date

import pandas as pd
import os
from pathlib import Path
from orjson import loads as json_loads

csv_path = Path('./all_csv_files').resolve()

DatasetsPaths = dict[str, list[tuple[date, Path]]]

users_required_columns = [
    'user_id', 'ts', 'ampl_user_data'
]

payments_required_columns = [
    'insert_id', 'user_id', 'ts', 'price_usd',
]

dataset_columns_map = {
    'users': users_required_columns,
    'payments': payments_required_columns,
    'messages': None
}


def get_dataset_names() -> DatasetsPaths:
    result = {
        'users': [],
        'messages': [],
        'payments': []
    }

    for file in csv_path.iterdir():
        if not file.is_file():
            continue

        name, ext = os.path.splitext(file.name)
        if ext != '.csv':
            continue

        if name.startswith('users_all_') or name.startswith('messages_all_') or name.startswith('payments_all_'):
            parts = name.split('_')
            dataset = parts[0]
            date_from = datetime.fromisoformat(parts[2])
            result[dataset].append((date_from, file))

    for dataset in result:
        result[dataset].sort(key=lambda x: x[0])

    return result


def ampl_user_data_bad_json_parse(value: str):
    result = {}

    for key, value in json_loads(value.replace("'", '"')).items():
        try:
            result[key.strip('$')] = bool(int(value))
        except (ValueError, TypeError):
            result[key.strip('$')] = None
    return result


def upload_dataset(prefix: str, dataset_path: Path, dataset_date: date, columns: list | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_path,
        usecols=columns if columns else None,
        converters={'ampl_user_data': ampl_user_data_bad_json_parse}
    )
    df['date'] = dataset_date
    return df


def upload_all_datasets(prefix: str, datasets: list[tuple[date, Path]], columns: list | None) -> pd.DataFrame:
    dfs = []
    for dataset_date, dataset_path in datasets:
        df = upload_dataset(prefix, dataset_path, dataset_date, columns)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def upload_all_users_datasets(datasets: list[tuple[date, Path]]) -> tuple[list[str], pd.DataFrame]:
    dfs = []
    experiments = set()
    for dataset_date, dataset_path in datasets:
        df = upload_dataset('users', dataset_path, dataset_date, dataset_columns_map['users'])
        exps, df = transform_users(df)
        experiments.update(exps)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    return list(experiments), df


def transform_users(users: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    try:
        df = pd.json_normalize(users['ampl_user_data'])
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        raise e

    df = pd.concat([users, df], axis=1)

    exp_cols = {col for col in df.columns if col.startswith('exp')}
    df = df.drop(columns=['ampl_user_data']).dropna(subset=list(exp_cols), how='all')
    df['ts'] = pd.to_datetime(df['ts'])
    shape, unique = df.shape[0], df['user_id'].nunique()
    df = df.sort_values(['user_id', 'ts']) # (40004, 12883)
    df = df.drop_duplicates(subset=['user_id'], keep='last')
    ashape, aunique = df.shape[0], df['user_id'].nunique()

    return exp_cols, df


def transform_payments(payments: pd.DataFrame) -> pd.DataFrame:
    payments['ts'] = pd.to_datetime(payments['ts'])
    payments = payments.sort_values(['ts', 'insert_id'])
    payments = payments.drop_duplicates(subset=['insert_id'], keep='first')

    payments['timedelta'] = payments['ts'].diff()
    filtered = payments[
        (payments['timedelta'].isna()) |
        (payments['timedelta'] >= pd.Timedelta(milliseconds=300))
    ]

    return filtered.drop(columns=['timedelta'])


def upload_and_merge_datasets(dataset_paths: DatasetsPaths) -> tuple[list[str], pd.DataFrame]:
    assert 'users' in dataset_paths
    assert 'messages' in dataset_paths
    assert 'payments' in dataset_paths

    experiments, users_df = upload_all_users_datasets(dataset_paths.pop('users'))

    messages = upload_all_datasets('messages', dataset_paths.pop('messages'), dataset_columns_map['messages'])
    payments = upload_all_datasets('payments', dataset_paths.pop('payments'), dataset_columns_map['payments'])
    payments = transform_payments(payments)

    merged = (
        users_df.merge(messages, on=['user_id', 'date'], how='left')
        .merge(payments, on=['user_id', 'date'], how='left')
        .fillna({
            'messages_count': 0,
            'price_usd': 0
        })
    )

    return experiments, merged
