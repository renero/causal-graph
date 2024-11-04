"""

This module contains functions to compute the Maximal Information Coefficient
between pairs of features in a dataframe. The MIC is a measure of the strength
of the linear or non-linear association between two variables. The MIC is
computed using the MINE statistics, which is a non-parametric method that
computes the MIC between two variables by estimating the mutual information
between them.

"""

from itertools import combinations

import numpy as np
import pandas as pd

from mlforge.progbar import ProgBar # type: ignore


def pairwise_mic(
        data: pd.DataFrame,
        alpha=0.6,
        c=15,
        to_return='mic',
        est="mic_approx",
        prog_bar=True,
        silent=False):
    """
    From a dataframe, compute the MIC for each pair of features. See
    https://github.com/minepy/minepy and https://github.com/minepy/mictools for
    more details.

    - [Reshef2016]	Yakir A. Reshef, David N. Reshef, Hilary K. Finucane and Pardis C.
    Sabeti and Michael Mitzenmacher. Measuring Dependence Powerfully and Equitably.
    Journal of Machine Learning Research, 2016.
    - [Matejka2017]	J. Matejka and G. Fitzmaurice. Same Stats, Different Graphs:
    Generating Datasets with Varied Appearance and Identical Statistics through
    Simulated Annealing. ACM SIGCHI Conference on Human Factors in Computing
    Systems, 2017.

    Arguments:
        data (DataFrame): A DF with continuous numerical values
        alpha (float): MINE MIC value for alpha
        c (int): MINE MIC value for c
        to_return (str): Either 'mic' or 'tic'.
        est (str): MINE MIC value for est. Default is est=”mic_approx” where the
            original MINE statistics will be computed, with est=”mic_e” the
            equicharacteristic matrix is is evaluated and MIC_e and TIC_e are returned.
        prog_bar (bool): whether to print the prog_bar or not.

    Returns:
        A dataframe with the MIC values between pairs.
    """
    assert est in ["mic_approx",
                   "mic_e"], "est must be either mic_approx or mic_e"

    # Check if 'minpy' package is installed
    try:
        from minepy import pstats
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install 'minepy' package to use the 'pairwise_mic' function")

    mic_p, tic_p = pstats(data.values.T, alpha=alpha, c=c, est=est)
    m = len(data.columns)
    mic, tic = np.ones((m, m)), np.ones((m, m))

    # pbar = tqdm(total=m*(m-1)/2, **
    #             tqdm_params("Computing MIC", prog_bar, silent=silent))
    pbar = ProgBar().start_subtask(m*(m-1)/2)

    # desc="Computing MIC", disable=not prog_bar,
    # position=1, leave=False)
    for i in range(m):
        for j in range(i+1, m):
            k = int(m*i - i*(i+1)/2 - i - 1 + j)
            mic[i, j] = mic_p[k]
            mic[j, i] = mic_p[k]
            tic[i, j] = tic_p[k]
            tic[j, i] = tic_p[k]
            pbar.update_subtask(1)

    if to_return == "tic":
        return tic
    return mic
