import numpy as np
import pandas as pd
from scipy import stats

from preparation import get_aggregated_a_b_groups


def mannwhitney_test_impl(
        experiment: str,
        metric: str,
        df: pd.DataFrame,
        alpha: float = 0.12,
        n_resamples: int = 10000,
):
    from calculate import TestResult
    rng = np.random.default_rng(8)

    a, b = get_aggregated_a_b_groups(df, experiment, metric)

    mw = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    p_mw = mw.pvalue
    u_statistic = mw.statistic

    len_a, len_b = len(a), len(b)
    a_12 = u_statistic / (len_a * len_b)
    ci_level = 1 - alpha

    boot_vals = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        a_s = rng.choice(a, size=len_a, replace=True)
        b_s = rng.choice(b, size=len_b, replace=True)
        u_s = stats.mannwhitneyu(a_s, b_s, alternative='two-sided', method='asymptotic').statistic
        boot_vals[i] = u_s / (len_a * len_b)
    ci_lo, ci_hi = np.percentile(boot_vals, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])

    direction = None
    if p_mw > alpha:
        decision = 'REJECT'
        reason = f'p value > alpha: {p_mw} > {alpha}; no stochastic difference'
    elif ci_hi < 0.5:
        decision = 'ACCEPT'
        reason = f'p value < alpha: {p_mw} < {alpha}; B is stochastically less then A'
        direction = '-'
    elif ci_lo > 0.5:
        decision = 'ACCEPT'
        reason = f'p value < alpha: {p_mw} < {alpha}; B is stochastically greater then A'
        direction = '+'
    else:
        decision = 'KEEP_RUNNING'
        reason = f'p value < alpha: {p_mw} < {alpha}; Not enough data to check statistically significant difference'
        direction = None

    return TestResult(
        experiment=experiment,
        test_name='mannwhithney',
        metric=metric,
        p_value=p_mw,
        ci=(ci_lo, ci_hi),
        decision=decision,
        direction=direction,
        reason=reason,
        vis_info={
            'resample_distribution': boot_vals,
            'a_12': a_12,
        }
    )
