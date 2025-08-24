import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats._resampling import BootstrapResult

from preparation import get_aggregated_a_b_groups


def bootstrap_test_impl(
        experiment: str,
        metric: str,
        df: pd.DataFrame,
        alpha: float = 0.12,
        n_resamples: int = 10000,
):
    from calculate import TestResult
    rng = np.random.default_rng(8)

    a, b = get_aggregated_a_b_groups(df, experiment, metric)
    delta = b.mean() - a.mean()

    def diff_means(x, y):
        return float(np.mean(y) - np.mean(x))

    result: BootstrapResult = stats.bootstrap(
        data=(a, b),
        statistic=diff_means,
        paired=False,
        confidence_level=1 - alpha,
        vectorized=False,
        n_resamples=n_resamples,
        method='percentile',
        rng=rng
    )
    # p_bt = float(np.mean(np.abs(result.bootstrap_distribution) >= abs(delta)))
    p_bt = float(np.mean(result.bootstrap_distribution >= delta))
    ci_lo, ci_hi = result.confidence_interval.low, result.confidence_interval.high

    direction = None
    if p_bt > alpha:
        decision = 'REJECT'
        reason = f'p value > alpha; {p_bt} > {alpha} no meaningful difference between averages'
    else:
        if ci_lo <= 0 <= ci_hi:
            decision = 'KEEP_RUNNING'
            reason = (f'p value < alpha; {p_bt} < {alpha}, but 0 is in CI ({ci_lo}, {ci_hi}), not sure about '
                      f'difference direction')
        else:
            decision = 'ACCEPT'
            reason = f'p value < alpha; {p_bt} < {alpha}; 0 is not in CI ({ci_lo}, {ci_hi})'
            if ci_hi < 0:
                direction = '-'
            else:
                direction = '+'

    return TestResult(
        experiment=experiment,
        test_name='bootstrap_test',
        metric=metric,
        p_value=p_bt,
        decision=decision,
        reason=reason,
        direction=direction,
        ci=(ci_lo, ci_hi),
        vis_info={
            'resample_distribution': result.bootstrap_distribution,
            'delta_hat': delta,
        }
    )
