import numpy as np
import pandas as pd

from scipy import stats


def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests." \
        Journal of the Royal Statistical Society: Series B (Methodological) 46.3 \
            (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """

    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z):
            try:
                c, _, d, _ = stats.chi2_contingency(
                    df.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
                )
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if not isinstance(z_state, str):
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return chi, p_value, dof


def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    r"""
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    :math:`P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as :math:`P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="pearson", **kwargs
    )


def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    r"""
    Computes Pearson correlation coefficient and p-value for testing non-correlation.
    Should be used only on continuous data. In case when :math:`Z != \null` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27C2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27C2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27C2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    Pearson's correlation coefficient: float
    p-value: float

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(
            f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 2: If Z is empty compute a non-conditional test.
    if len(Z) == 0:
        coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

    # Step 3: If Z is non-empty, use linear regression to compute residuals and test independence on it.
    else:
        X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
        Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

        residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
        residual_Y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)
        coef, p_value = stats.pearsonr(residual_X, residual_Y)

    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value
