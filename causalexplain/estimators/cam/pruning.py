"""

- The `pruning` function is translated to Python.
- `dim(G)[1]` is replaced with `G.shape[0]` to get the number of rows.
- `matrix(0,p,p)` is replaced with `np.zeros((p, p))` to create a zero matrix.
- `which(G[,i]==1)` is replaced with `np.where(G[:, i] == 1)[0]` to find the indices
    where the condition is true.
- `cbind(X[,parents],X[,i])` is replaced with `np.hstack((X[:, parents], X[:, [i]]))`
    to concatenate arrays horizontally.
- The `cat` function is replaced with `print` for output.
- The `pruneMethod` function is passed as `prune_method` and called accordingly.
"""
import numpy as np

from .selGam import selGam


def pruning(
        X,
        G,
        verbose=False,
        prune_method=None,
        prune_method_pars={'cutOffPVal': 0.001, 'numBasisFcts': 10}):
    """_summary_

    Args:
        X (_type_): Input vectors
        G (_type_): Adjacency matrix representing a DAG
        output (bool, optional): Whether to print debug messages
        prune_method (_type_, optional): _description_. Defaults to None.
        prune_method_pars (dict, optional): _description_.
            Defaults to {'cutOffPVal': 0.001, 'numBasisFcts': 10}.

    Returns:
        _type_: _description_
    """
    if prune_method is None:
        prune_method = selGam

    p = G.shape[0]
    finalG = np.zeros((p, p))

    for i in range(p):
        parents = np.where(G[:, i] == 1)[0]
        lenpa = len(parents)

        if verbose:
            print(f"Pruning variable: {i}")
            print(f". Considered parents: {parents}")

        if lenpa > 0:
            Xtmp = np.hstack((X[:, parents], X[:, [i]]))
            selected_par = prune_method(
                Xtmp, pars=prune_method_pars, verbose=verbose, k=lenpa+1)
            final_parents = parents[selected_par]
            finalG[final_parents, i] = 1
            if verbose:
                print(f". Final parents: {final_parents}")
                print(f". Pruned parents of {i}: {[p for p in parents if p not in final_parents]}")

    return finalG
