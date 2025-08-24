import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats._stats_py import TtestResult
from statsmodels.stats.power import TTestIndPower

from preparation import get_aggregated_a_b_groups, bootstrap_resample

effect_sizes = {
    'arpu': 0.5,
    'messages': 5,
    'user_retention': 0.08,
}


def calculate_sufficient_sample_groups(a, b, alpha: float, power: float, metric: str) -> tuple[bool, int]:
    analysis = TTestIndPower()
    effect_size = effect_sizes[metric]
    effect_size /= np.std(a)

    n_per_group = np.ceil(
        analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=len(b) / len(a), alternative='two-sided'))

    return len(a) >= n_per_group and len(b) >= n_per_group, n_per_group


def t_test_impl(
        experiment: str,
        metric: str,
        df: pd.DataFrame,
        alpha: float = 0.12,
        n_resamples: int = 10000,
):
    from calculate import TestResult
    rng = np.random.default_rng(8)

    a, b = get_aggregated_a_b_groups(df, experiment, metric)

    is_sufficient, n_per_group = calculate_sufficient_sample_groups(a, b, alpha, 0.8, metric)
    decision = None
    reason = ''
    if not is_sufficient:
        decision = 'KEEP_RUNNING'
        reason = f'not sufficient group sizes; group sizes a={len(a)} b={len(b)}; required sample size: {n_per_group}'
        print(f'Warning! In t-test of {experiment}-{metric} there are {reason}')

    n_resamples = max(n_resamples, n_per_group)

    mean_bootstrap_a, mean_bootstrap_b = bootstrap_resample(a, b, alpha, n_resamples, np.mean, rng)
    mean_bootstrap_a, mean_bootstrap_b = mean_bootstrap_a.bootstrap_distribution, mean_bootstrap_b.bootstrap_distribution

    test_result: TtestResult = ttest_ind(
        mean_bootstrap_b,
        mean_bootstrap_a,
        alternative='two-sided',
    )

    p_tt = test_result.pvalue
    ci_lo, ci_hi = test_result.confidence_interval(1 - alpha)

    direction = None
    if p_tt > alpha:
        decision = decision or 'REJECT'
        reason = reason or f'p value > alpha; {p_tt} > {alpha} no meaningful difference between averages'
    else:
        if ci_lo <= 0 <= ci_hi:
            decision = 'KEEP_RUNNING'
            l_reason = (f'p value < alpha; {p_tt} < {alpha}, but 0 is in CI ({ci_lo}, {ci_hi}), not sure about '
                        'difference direction')
            if reason:
                reason += '; ' + l_reason
            else:
                reason = l_reason
        else:
            decision = decision or 'ACCEPT'
            l_reason = f'p value < alpha; {p_tt} < {alpha}; 0 is not in CI ({ci_lo}, {ci_hi})'
            if reason:
                reason += '; ' + l_reason
            else:
                reason = l_reason

            if ci_hi < 0:
                direction = '-'
            else:
                direction = '+'

    return TestResult(
        experiment=experiment,
        test_name='ttest',
        metric=metric,
        p_value=p_tt,
        decision=decision,
        reason=reason,
        direction=direction,
        ci=(ci_lo, ci_hi),
        vis_info={
            'resample_distribution': test_result.statistic,
            'delta_hat': mean_bootstrap_b.mean() - mean_bootstrap_a.mean(),
            'perm_null': mean_bootstrap_b - mean_bootstrap_a,
        }
    )
