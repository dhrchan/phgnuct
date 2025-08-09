import numpy as np
from scipy import stats


def two_sample_ttest(mean1, mean2, stdev1, stdev2, nsamples1, nsamples2):
    """
    Calculate two-sample t-test p-value assuming unequal variances (Welch's t-test).

    Parameters:
    -----------
    mean1 : float
        Mean of sample 1
    mean2 : float
        Mean of sample 2
    stdev1 : float
        Standard deviation of sample 1
    stdev2 : float
        Standard deviation of sample 2
    nsamples1 : int
        Number of samples in group 1
    nsamples2 : int
        Number of samples in group 2

    Returns:
    --------
    p_value : float
        Two-tailed p-value of Welch's t-test
    """

    # Calculate variances
    var1 = stdev1**2
    var2 = stdev2**2

    # Calculate difference in means
    mean_diff = mean1 - mean2

    # Calculate standard error (unequal variances)
    standard_error = np.sqrt(var1 / nsamples1 + var2 / nsamples2)

    # Calculate degrees of freedom using Welch-Satterthwaite equation
    df = (var1 / nsamples1 + var2 / nsamples2) ** 2 / (
        (var1 / nsamples1) ** 2 / (nsamples1 - 1)
        + (var2 / nsamples2) ** 2 / (nsamples2 - 1)
    )

    # Calculate t-statistic
    t_stat = mean_diff / standard_error

    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return p_value

