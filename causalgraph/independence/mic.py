from deprecated import deprecated
import numpy as np
import pandas as pd

from itertools import combinations
from minepy import MINE, pstats
from tqdm.auto import tqdm



def pairwise_mic(
    data: pd.DataFrame, 
    alpha=0.6, 
    c=15,
    to_return='mic',
    est="mic_approx", 
    progbar=True):
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
        progbar (bool): whether to print the progbar or not.

    Returns:
        A dataframe with the MIC values between pairs.
    """
    assert est in ["mic_approx", "mic_e"], "est must be either mic_approx or mic_e"

    mic_p, tic_p = pstats(data.values.T, alpha=alpha, c=c, est=est)
    m = len(data.columns)
    mic, tic = np.ones((m, m)), np.ones((m, m))
    pbar = tqdm(total=m*(m-1)/2, desc="Computing MIC", disable=not progbar, leave=False)
    for i in range(m):
        for j in range(i+1, m):
            pbar.refresh()
            k = int(m*i - i*(i+1)/2 - i - 1 + j)
            mic[i, j] = mic_p[k]
            mic[j, i] = mic_p[k]
            tic[i, j] = tic_p[k]
            tic[j, i] = tic_p[k]
            pbar.update(1)
    pbar.close()
    if to_return == "tic":
        return tic
    return mic


@deprecated("Use pairwise_mic instead")
def pairwise_MIC(data: pd.DataFrame, c=15, alpha=0.6, progbar=True):
    """
    From a dataframe, compute the MIC for each pair of features.

    Arguments:
        data (DataFrame): A DF with continuous numerical values
        c (int): MINE MIC value for c
        alpha (float): MINE MIC value for alpha
        progbar (bool): whether to print the progbar or not.

    Returns:
        A dataframe with the MIC values between pairs.
    """
    list_nodes = list(data.columns.values)
    list_pairs = list(combinations(list_nodes, 2))
    mine = MINE(alpha=alpha, c=c)
    score_df = pd.DataFrame(0, index=data.columns, columns=data.columns)
    pbar = tqdm(total=len(list_pairs),
                disable=not progbar, desc="Computing MIC", leave=False)
    for feat1, feat2 in list_pairs:
        pbar.update(1)
        x, y = data[feat1], data[feat2]
        mine.compute_score(x, y)
        coef = mine.mic()
        score_df.loc[feat1, feat2] = coef
        score_df.loc[feat2, feat1] = coef
        pbar.refresh()
    pbar.close()
    np.fill_diagonal(score_df.values, 1.0)
    return score_df
