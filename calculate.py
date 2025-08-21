import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd
import pebble

from mannwhitney_test import mannwhitney_test_impl
from permutation_test import permutation_test_impl
from preparation import prepare_for_experiment


@dataclass
class TestResult:
    test_name: str
    experiment: str
    metric: str
    p_value: float

    ci: tuple[float, float] | None = None

    decision: str | None = None
    direction: str | None = None
    reason: str | None = None
    vis_info: dict[str, Any] | None = None


tests = {
    # 't-test': t_test,
    # 'z_test': z_test,
    'permutation': permutation_test_impl,
    'mannwhitney': mannwhitney_test_impl
}


def run_test(args):
    test_name, metric, experiment, df = args

    assert test_name in tests, f'{test_name} not in tests'

    test = tests[test_name]

    try:
        result = test(experiment, metric, df)
    except Exception as e:
        print(f'error in {test_name}: {e}')
        raise e

    return result


def run_experiments(df: pd.DataFrame, experiments: list[str]) -> dict[str, dict[str, list[TestResult]]]:
    tasks = [
        (experiment, prepare_for_experiment(df, experiment))
        for experiment in experiments[:]
    ]
    tasks = [
        (metric, experiment, df)
        for metric, (experiment, df) in product(['arpu', 'messages'], tasks)
    ]

    tasks = [
        (test, metric, experiment, df)
        for test, (metric, experiment, df) in product(tests.keys(), tasks)
    ]

    tasks = [
        tuple(task)
        for task in tasks
    ]
    results: dict[str, dict[str, list[TestResult]]] = defaultdict(lambda: defaultdict(list))

    with pebble.ProcessPool(min(len(tasks), os.cpu_count())) as pool:
        map_future = pool.map(run_test, tasks)

        for test_result in map_future.result():
            test_result: TestResult | None = test_result
            if not test_result:
                continue
            results[test_result.experiment][test_result.metric].append(test_result)

    for experiment, experiment_test_results in results.items():
        print(f'####### {experiment} ####### ')
        results_df = pd.DataFrame(columns=['test', 'metric', 'decision', 'direction', 'reason'])

        for metric, metric_tests_results in experiment_test_results.items():
            for test_result in sorted(metric_tests_results, key=lambda x: x.test_name):
                row = {
                    'test': test_result.test_name,
                    'metric': metric,
                    'decision': test_result.decision,
                    'direction': test_result.direction,
                    'reason': test_result.reason
                }
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        decisions = results_df['decision'].unique()
        if len(decisions) == 1 and decisions[0] == 'REJECT':
            print(f'Experiment {experiment} should be REJECTED; both metrics are rejected')
        elif 'ACCEPT' not in decisions:
            print(f'Experiment `{experiment}` should KEEP RUNNING:')
            exps_to_keep_running = results_df[['test', 'metric', 'reason']][
                results_df['decision'] == 'KEEP_RUNNING'].to_numpy()

            for test, metric, reason in exps_to_keep_running.values():
                print(f'Test {test} for metric {metric} should KEEP RUNNING because\n- {reason}')
        else:
            directions = results_df.dropna()['direction'].unique()
            accepted_tests = results_df[['test', 'metric', 'direction', 'reason']][
                results_df['decision'] == 'ACCEPT'].to_numpy()
            if len(directions) == 1:
                if directions[0] == '-':
                    for test, metric, direction, reason in accepted_tests:
                        print(f'Test {test} for metric {metric} is negative\n- {reason}')
                    print(f'Experiment `{experiment}` should be ACCEPTED, even if effect is negative')
                else:
                    for test, metric, direction, reason in accepted_tests:
                        print(f'Test {test} for metric {metric} has positive effect\n- {reason}')
                    print(f'Experiment `{experiment}` should be ACCEPTED')
            else:
                print(f'Experiment `{experiment}` should be ACCEPTED, but there are negative and positive effects')
                for test, metric, direction, reason in accepted_tests:
                    direction = 'negative' if direction == '-' else 'positive'
                    print(f'Test {test} for metric {metric} is {direction}\n- {reason}')
                print(
                    f'Should ask someone how it will impact revenue '
                    '(example: a lot of messages, but users tend to not to spend money '
                    'which means more money is spent for servers and employees)'
                )
    print('Done!')

    return results
