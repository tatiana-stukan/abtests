import numpy as np
import pandas as pd
from scipy import stats

from preparation import get_aggregated_a_b_groups


def permutation_test_impl(
        experiment: str,
        metric: str,
        df: pd.DataFrame,
        alpha: float = 0.12,
        n_resamples: int = 10000,
):
    from calculate import TestResult
    rng = np.random.default_rng(8)

    a, b = get_aggregated_a_b_groups(df, experiment, metric)
    len_a, len_b = len(a), len(b)

    def diff_means(x, y):
        return float(np.mean(y) - np.mean(x))

    perm_res = stats.permutation_test(
        data=(a, b),
        statistic=diff_means,
        permutation_type='independent',
        alternative='two-sided',
        n_resamples=n_resamples,
        vectorized=False,
        rng=rng,
    )
    p_perm = float(perm_res.pvalue)

    conf_interval = stats.bootstrap(
        data=(a, b),
        statistic=diff_means,
        paired=False,
        confidence_level=1 - alpha,
        vectorized=False,
        n_resamples=n_resamples,
        method='percentile',
        rng=rng
    )

    direction = None
    if p_perm > alpha:
        decision = 'REJECT'
        reason = f'p value > alpha; {p_perm} > {alpha} no meaningful difference between averages'
    else:
        ci_lo = conf_interval.confidence_interval.low
        ci_hi = conf_interval.confidence_interval.high
        if ci_lo <= 0 <= ci_hi:
            decision = 'KEEP_RUNNING'
            reason = (f'p value < alpha; {p_perm} < {alpha}, but 0 is in CI ({ci_lo}, {ci_hi}), not sure about '
                      f'difference direction')
        else:
            decision = 'ACCEPT'
            reason = f'p value < alpha; {p_perm} < {alpha}; 0 is not in CI ({ci_lo}, {ci_hi})'
            if ci_hi < 0:
                direction = '-'
            else:
                direction = '+'

    return TestResult(
        experiment=experiment,
        test_name='permutation',
        metric=metric,
        p_value=p_perm,
        decision=decision,
        reason=reason,
        direction=direction,
        ci=conf_interval.confidence_interval,
        vis_info={
            'resample_distribution': conf_interval.bootstrap_distribution,
            'delta_hat': b.mean() - a.mean(),
            'perm_null': perm_res.null_distribution
        }
    )
