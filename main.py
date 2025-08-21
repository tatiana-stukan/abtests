from calculate import TestResult


def run_all_tests() -> dict[str, dict[str, list[TestResult]]]:
    from calculate import run_experiments
    from upload_datasets import upload_and_merge_datasets, get_dataset_names

    experiments, df = upload_and_merge_datasets(get_dataset_names())

    return run_experiments(df, experiments)


if __name__ == '__main__':
    run_all_tests()
