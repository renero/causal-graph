# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


"""The main module, containing the implementation of GES, including
the logic for the insert, delete and turn operators. The
implementation is directly based on two papers:

  1. The 2002 GES paper by Chickering, "Optimal Structure
  Identification With Greedy Search" -
  https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

  2. For the turn operator, the 2012 GIES paper by Hauser & Bühlmann,
  "Characterization and Greedy Learning of Interventional Markov
  Equivalence Classes of Directed Acyclic Graphs" -
  https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf

Further credit is given where due.

Additional modules / packages:

  - ges.utils contains auxiliary functions and the logic to transform
    a PDAG into a CPDAG, used after each application of an operator.
  - ges.scores contains the modules with the score classes:
      - ges.scores.decomposable_score contains the base class for
        decomposable score classes (see that module for more details).
      - ges.scores.gauss_obs_l0_pen contains a cached implementation
        of the gaussian likelihood BIC score used in the original GES
        paper.
   - ges.test contains the modules with the unit tests and tests
     comparing against the algorithm's implementation in the R package
     'pcalg'.

"""

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.discriminant_analysis import StandardScaler

from causalgraph.estimators.ges.scores import GaussObsL0Pen
from causalgraph.estimators.ges import utils
from ...common.utils import graph_from_adjacency, graph_from_dot_file
from causalgraph.metrics.compare_graphs import evaluate_graph


class GES(object):
    """
    The GES (Greedy Equivalence Search) algorithm for learning causal
    graphs from observational data.

    Parameters
    ----------
    phases : list, optional
        Which phases of the GES procedure are run, and in which order.
        Defaults to `['forward', 'backward', 'turning']`.
    iterate : bool, default=False
        Indicates whether the given phases should be iterated more than once.
    debug : int, optional
        Debug level for printing traces. Higher values correspond to increased
        verbosity.

    Attributes
    ----------
    dag : numpy.ndarray
        The adjacency matrix of the estimated CPDAG (completed partially
        directed acyclic graph).
    ges_adjmat : numpy.ndarray
        The adjacency matrix of the estimated PDAG (partially directed acyclic
        graph) before completion.
    ges_score : float
        The score of the estimated CPDAG.
    phases : list
        The phases of the GES procedure that are run, and in which order.
    iterate : bool
        Indicates whether the given phases should be iterated more than once.

    Methods
    -------
    fit(data, labels=None, A0=None)
        Fit the GES algorithm to the given data.
    fit_bic(data, A0=None, iterate=False, debug=0)
        Run GES on the given data using the Gaussian BIC score.
    forward_step(A, cache, debug=0)
        Score and apply the highest scoring insert operator.
    backward_step(A, cache, debug=0)
        Score and apply the highest scoring delete operator.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> from causalgraph.estimators.ges import GES  # doctest: +SKIP
    >>> data = np.array([[3.23125779, 3.24950062, 13.430682, 24.67939513],  # doctest: +SKIP
    ...                  [1.90913354, -0.06843781, 6.93957057, 16.10164608],# doctest: +SKIP
    ...                  [2.68547149, 1.88351553, 8.78711076, 17.18557716], # doctest: +SKIP
    ...                  [0.16850822, 1.48067393, 5.35871419, 11.82895779], # doctest: +SKIP
    ...                  [0.07355872, 1.06857039, 2.05006096, 3.07611922]]) # doctest: +SKIP
    >>> ges = GES() # doctest: +SKIP
    >>> ges.fit(data)   # doctest: +SKIP
    """

    dag = None
    ges_adjmat = None
    ges_score = None
    phases = None
    iterate = None
    debug = 0
    is_fitted_ = False
    feature_names = None
    metrics = None

    def __init__(
            self,
            name: str,
            phases=None,
            iterate=False,
            debug=0,
            verbose=False):
        self.name = name
        self.phases = [
            'forward', 'backward', 'turning'
        ] if phases is None else phases
        self.iterate = iterate
        self.debug = int(verbose)

    def fit(self, X, A0=None):
        assert isinstance(X, pd.DataFrame), "Data must be a pandas DataFrame"
        self.feature_names = list(X.columns)
        X = X.values
        self.ges_adjmat, self.ges_score = self.fit_bic(X, A0=A0)
        self.dag = graph_from_adjacency(
            self.ges_adjmat, node_labels=self.feature_names)
        self.is_fitted_ = True
        return self

    def fit_predict(self, train, test, ref_graph: nx.DiGraph = None, A0=None):
        """
        Fit the GES algorithm to the given data and return the
        adjacency matrix of the estimated CPDAG.

        Parameters
        ----------
        data : numpy.ndarray
            The n x p array containing the observations, where columns
            correspond to variables and rows to observations.
        labels : list, optional
            The list of variable names. If not given, the columns of
            data are used as labels.
        A0 : numpy.ndarray, optional
            The initial CPDAG on which GES will run, where where `A0[i,j]
            != 0` implies the edge `i -> j` and `A[i,j] != 0 & A[j,i] !=
            0` implies the edge `i - j`. Defaults to the empty graph
            (i.e. matrix of zeros).

        Returns
        -------
        estimate : numpy.ndarray
            The adjacency matrix of the estimated CPDAG.

        Raises
        ------
        TypeError:
            If the type of some of the parameters was not expected,
            e.g. if data is not a numpy array.
        ValueError:
            If the value of some of the parameters is not appropriate,
            e.g. a wrong phase is specified.
        """
        self.fit(train, A0)
        if ref_graph:
            self.metrics = evaluate_graph(
                ref_graph, self.dag, self.feature_names)
        return self.dag

    def fit_bic(self, data, A0=None):
        """
        Run GES on the given data, using the Gaussian BIC score
        (l0-penalized Gaussian Likelihood). The data is not assumed to be
        centered, i.e. an intercept is fitted.

        To use a custom score, see ges.fit.

        Parameters
        ----------
        data : numpy.ndarray
            The n x p array containing the observations, where columns
            correspond to variables and rows to observations.
        A0 : numpy.ndarray, optional
            The initial CPDAG on which GES will run, where where `A0[i,j]
            != 0` implies the edge `i -> j` and `A[i,j] != 0 & A[j,i] !=
            0` implies the edge `i - j`. Defaults to the empty graph
            (i.e. matrix of zeros).

        Returns
        -------
        estimate : numpy.ndarray
            The adjacency matrix of the estimated CPDAG.
        total_score : float
            The score of the estimate.

        Raises
        ------
        TypeError:
            If the type of some of the parameters was not expected,
            e.g. if data is not a numpy array.
        ValueError:
            If the value of some of the parameters is not appropriate,
            e.g. a wrong phase is specified.

        Example
        -------
        Data from a linear-gaussian SCM (generated using
        `sempler <https://github.com/juangamella/sempler>`__)

        >>> import numpy as np
        >>> data = np.array([[3.23125779, 3.24950062, 13.430682, 24.67939513],
        ...                  [1.90913354, -0.06843781, 6.93957057, 16.10164608],
        ...                  [2.68547149, 1.88351553, 8.78711076, 17.18557716],
        ...                  [0.16850822, 1.48067393, 5.35871419, 11.82895779],
        ...                  [0.07355872, 1.06857039, 2.05006096, 3.07611922]])

        Run GES using the gaussian BIC score:

        >>> from causalgraph.estimators.ges import GES  # doctest: +SKIP
        >>> ges = GES()                                 # doctest: +SKIP
        >>> ges.fit_bic(data)                           # doctest: +SKIP
        (array([[0, 1, 1, 0],                           # doctest: +SKIP
            [0, 0, 0, 0],                               # doctest: +SKIP
            [1, 1, 0, 1],                               # doctest: +SKIP
            [0, 1, 1, 0]]), 15.674267611628233)         # doctest: +SKIP
        """
        # Initialize Gaussian BIC score (precomputes scatter matrices, sets up cache)
        cache = GaussObsL0Pen(data)
        # Unless indicated otherwise, initialize to the empty graph
        A0 = np.zeros((cache.p, cache.p)) if A0 is None else A0
        return self._fit(cache, None, A0)

    def _fit(
            self,
            score_class,
            completion_algorithm=None,
            A0=None):
        """
        Run GES using a user defined score.

        Parameters
        ----------
        score_class : ges.DecomposableScore
            an instance of a class which inherits from
            ges.decomposable_score.DecomposableScore (or defines a
            local_score function and a p attribute, see
            ges.decomposable_score for more info).
        completion_algorithm : function, optional
            the "completion algorithm" used to go from PDAG to CPDAG after
            the application of each operator. If `None`, the algorithm
            used in the original GES paper is used.
        A0 : np.array, optional
            the initial CPDAG on which GES will run, where where A0[i,j]
            != 0 implies i -> j and A[i,j] != 0 & A[j,i] != 0 implies i -
            j. Defaults to the empty graph.

        Returns
        -------
        estimate : np.array
            the adjacency matrix of the estimated CPDAG
        total_score : float
            the score of the estimate

        Raises
        ------
        ValueError
            If no phase is specified.
        """
        # Select the desired phases
        if len(self.phases) == 0:
            raise ValueError("Must specify at least one phase")
        # Set the completion algorithm
        completion_algorithm = utils.pdag_to_cpdag if completion_algorithm is None else completion_algorithm
        # Unless indicated otherwise, initialize to the empty graph
        A0 = np.zeros((score_class.p, score_class.p)) if A0 is None else A0
        # GES procedure
        total_score = 0
        A, score_change = A0, np.Inf
        # Run each phase
        while True:
            last_total_score = total_score
            for phase in self.phases:
                if phase == 'forward':
                    fun = self.forward_step
                elif phase == 'backward':
                    fun = self.backward_step
                elif phase == 'turning':
                    fun = self.turning_step
                else:
                    raise ValueError(f'Invalid phase "{phase}" specified')
                print(f"\nGES {phase} phase start\n"
                      f"-------------------------") if self.debug else None
                while True:
                    score_change, new_A = fun(
                        A, score_class, max(0, self.debug - 1))
                    if score_change > 0:
                        A = completion_algorithm(new_A)
                        total_score += score_change
                    else:
                        break
                print("-----------------------") if self.debug else None
                print("GES %s phase end" % phase) if self.debug else None
                print("Total score: %0.4f" %
                      total_score) if self.debug else None
                [print(row) for row in A] if self.debug else None
            if total_score <= last_total_score or not self.iterate:
                break
        return A, total_score

    def forward_step(self, A, cache, debug=0):
        """
        Scores all valid insert operators that can be applied to the current
        CPDAG A, and applies the highest scoring one.

        Parameters
        ----------
        A : np.array
            the adjacency matrix of a CPDAG, where A[i,j] != 0 => i -> j
            and A[i,j] != 0 & A[j,i] != 0 => i - j.
        cache : DecomposableScore
            an instance of the score class, which computes the change in
            score and acts as cache.
        debug : int, optional
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        score : float
            the change in score resulting from applying the highest
            scoring operator (note, can be smaller than 0).
        new_A : np.array
            the adjacency matrix of the PDAG resulting from applying the
            operator (not yet a CPDAG).
        """
        # Construct edge candidates (i.e. edges between non-adjacent nodes)
        fro, to = np.where((A + A.T + np.eye(len(A))) == 0)
        edge_candidates = list(zip(fro, to))
        # For each edge, enumerate and score all valid operators
        valid_operators = []
        print("  %d candidate edges" %
              len(edge_candidates)) if debug > 1 else None
        for (x, y) in edge_candidates:
            valid_operators += self.score_valid_insert_operators(
                x, y, A, cache, debug=max(0, debug - 1))
        # Pick the edge/operator with the highest score
        if len(valid_operators) == 0:
            print("  No valid insert operators remain") if debug else None
            return 0, A
        else:
            scores = [op[0] for op in valid_operators]
            score, new_A, x, y, T = valid_operators[np.argmax(scores)]
            if debug:
                print(
                    f"  Best operator: insert({x}, {y}, {T}) -> ({score:.4f})")
            return score, new_A

    def backward_step(self, A, cache, debug=0):
        """
        Scores all valid delete operators that can be applied to the current
        CPDAG A, and applies the highest scoring one.

        Parameters
        ----------
        A : np.array
            the adjacency matrix of a CPDAG, where A[i,j] != 0 => i -> j
            and A[i,j] != 0 & A[j,i] != 0 => i - j.
        cache : DecomposableScore
            an instance of the score class, which computes the change in
            score and acts as cache.
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        score : float
            the change in score resulting from applying the highest
            scoring operator (note, can be smaller than 0).
        new_A : np.array
            the adjacency matrix of the PDAG resulting from applying the
            operator (not yet a CPDAG).
        """
        # Construct edge candidates:
        #   - directed edges
        #   - undirected edges, counted only once
        fro, to = np.where(utils.only_directed(A))
        directed_edges = zip(fro, to)
        fro, to = np.where(utils.only_undirected(A))
        undirected_edges = filter(
            lambda e: e[0] > e[1], zip(fro, to))  # zip(fro,to)
        edge_candidates = list(directed_edges) + list(undirected_edges)
        assert len(edge_candidates) == utils.skeleton(A).sum() / 2
        # For each edge, enumerate and score all valid operators
        valid_operators = []
        print("  %d candidate edges" %
              len(edge_candidates)) if debug > 1 else None
        for (x, y) in edge_candidates:
            valid_operators += self.score_valid_delete_operators(
                x, y, A, cache, debug=max(0, debug - 1))
        # Pick the edge/operator with the highest score
        if len(valid_operators) == 0:
            print("  No valid delete operators remain") if debug else None
            return 0, A
        else:
            scores = [op[0] for op in valid_operators]
            score, new_A, x, y, H = valid_operators[np.argmax(scores)]
            if debug:
                print(
                    f"  Best operator: delete({x}, {y}, {H}) -> ({score:.4f})")
            return score, new_A

    def turning_step(self, A, cache, debug=0):
        """
        Scores all valid turn operators that can be applied to the current
        CPDAG A, and applies the highest scoring one.

        Parameters
        ----------
        A : np.array
            the adjacency matrix of a CPDAG, where A[i,j] != 0 => i -> j
            and A[i,j] != 0 & A[j,i] != 0 => i - j.
        cache : DecomposableScore
            an instance of the score class, which computes the change in
            score and acts as cache.
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        score : float
            the change in score resulting from applying the highest
            scoring operator (note, can be smaller than 0).
        new_A : np.array
            the adjacency matrix of the PDAG resulting from applying the
            operator (not yet a CPDAG).

        """
        # Construct edge candidates:
        #   - directed edges, reversed
        #   - undirected edges
        fro, to = np.where(A != 0)
        edge_candidates = list(zip(to, fro))
        # For each edge, enumerate and score all valid operators
        valid_operators = []
        if debug > 1:
            print(f"  {len(edge_candidates)} candidate edges")
        for (x, y) in edge_candidates:
            valid_operators += self.score_valid_turn_operators(x, y, A, cache,
                                                               debug=max(0, debug - 1))
        # Pick the edge/operator with the highest score
        if len(valid_operators) == 0:
            print("  No valid turn operators remain") if debug else None
            return 0, A
        else:
            scores = [op[0] for op in valid_operators]
            score, new_A, x, y, C = valid_operators[np.argmax(scores)]
            if debug:
                print(f"  Best operator: turn({x}, {y}, {C}) -> ({score:.4f})")
            return score, new_A

    def insert(self, x, y, T, A):
        # --------------------------------------------------------------------
        # Insert operator
        #    1. definition in function insert
        #    2. enumeration logic (to enumerate and score only valid
        #    operators) function in score_valid_insert_operators
        """
        Applies the insert operator:
        1) adds the edge x -> y
        2) for all t in T, orients the previously undirected edge t -> y

        Parameters
        ----------
        x : int
            the origin node (i.e. x -> y)
        y : int
            the target node
        T : iterable of ints
            a subset of the neighbors of y which are not adjacent to x
        A : np.array
            the current adjacency matrix

        Returns
        -------
        new_A : np.array
            the adjacency matrix resulting from applying the operator

        """
        # Check inputs
        T = sorted(T)
        if A[x, y] != 0 or A[y, x] != 0:
            raise ValueError("x=%d and y=%d are already connected" % (x, y))
        if len(T) == 0:
            pass
        elif not (A[T, y].all() and A[y, T].all()):
            raise ValueError(
                "Not all nodes in T=%s are neighbors of y=%d" % (T, y))
        elif A[T, x].any() or A[x, T].any():
            raise ValueError(
                "Some nodes in T=%s are adjacent to x=%d" % (T, x))
        # Apply operator
        new_A = A.copy()
        # Add edge x -> y
        new_A[x, y] = 1
        # Orient edges t - y to t -> y, for t in T
        new_A[T, y] = 1
        new_A[y, T] = 0
        return new_A

    def score_valid_insert_operators(self, x, y, A, cache, debug=0):
        """Generate and score all valid insert(x,y,T) operators involving the edge
        x-> y, and all possible subsets T of neighbors of y which
        are NOT adjacent to x.

        Parameters
        ----------
        x : int
            the origin node (i.e. x -> y)
        y : int
            the target node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        p = len(A)
        if A[x, y] != 0 or A[y, x] != 0:
            raise ValueError("x=%d and y=%d are already connected" % (x, y))
        # One-hot encode all subsets of T0, plus one extra column to mark
        # if they pass validity condition 2 (see below)
        T0 = sorted(utils.neighbors(y, A) - utils.adj(x, A))
        if len(T0) == 0:
            subsets = np.zeros((1, p + 1), dtype=bool)
        else:
            subsets = np.zeros((2 ** len(T0), p + 1), dtype=bool)
            subsets[:, T0] = utils.cartesian([np.array([False, True])] * len(T0),
                                             dtype=bool)
        valid_operators = []
        print("    insert(%d,%d) T0=" % (x, y), set(T0)) if debug > 1 else None
        while len(subsets) > 0:
            print("      len(subsets)=%d, len(valid_operators)=%d" %
                  (len(subsets), len(valid_operators))) if debug > 1 else None
            # Access the next subset
            T = np.where(subsets[0, :-1])[0]
            passed_cond_2 = subsets[0, -1]
            subsets = subsets[1:]
            # Check that the validity conditions hold for T
            na_yxT = utils.na(y, x, A) | set(T)
            # Condition 1: Test that NA_yx U T is a clique
            cond_1 = utils.is_clique(na_yxT, A)
            if not cond_1:
                # Remove from consideration all other sets T' which
                # contain T, as the clique condition will also not hold
                supersets = subsets[:, T].all(axis=1)
                subsets = utils.delete(subsets, supersets, axis=0)
            # Condition 2: Test that all semi-directed paths from y to x contain a
            # member from NA_yx U T
            if passed_cond_2:
                # If a subset of T satisfied condition 2, so does T
                cond_2 = True
            else:
                # Check condition 2
                cond_2 = True
                for path in utils.semi_directed_paths(y, x, A):
                    if len(na_yxT & set(path)) == 0:
                        cond_2 = False
                        break
                if cond_2:
                    # If condition 2 holds for NA_yx U T, then it holds for all supersets of T
                    supersets = subsets[:, T].all(axis=1)
                    subsets[supersets, -1] = True
            print("      insert(%d,%d,%s)" % (x, y, T), "na_yx U T = ",
                  na_yxT, "validity:", cond_1, cond_2) if debug > 1 else None
            # If both conditions hold, apply operator and compute its score
            if cond_1 and cond_2:
                # Apply operator
                new_A = self.insert(x, y, T, A)
                # Compute the change in score
                aux = na_yxT | utils.pa(y, A)
                old_score = cache.local_score(y, aux)
                new_score = cache.local_score(y, aux | {x})
                print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" %
                      (y, aux | {x}, new_score, y, aux, old_score)) if debug > 1 else None
                # Add to the list of valid operators
                valid_operators.append((new_score - old_score, new_A, x, y, T))
                print("    insert(%d,%d,%s) -> %0.16f" %
                      (x, y, T, new_score - old_score)) if debug else None
        # Return all the valid operators
        return valid_operators

    def delete_operator(self, x, y, H, A):
        # --------------------------------------------------------------------
        # Delete operator
        #    1. definition in function delete
        #    2. enumeration logic (to enumerate and score only valid
        #    operators) function in valid_delete_operators
        """
        Applies the delete operator:
        1) deletes the edge x -> y or x - y
        2) for every node h in H
            * orients the edge y -> h
            * if the edge with x is undirected, orients it as x -> h

        Note that H must be a subset of the neighbors of y which are
        adjacent to x. A ValueError exception is thrown otherwise.

        Parameters
        ----------
        x : int
            the "origin" node (i.e. x -> y or x - y)
        y : int
            the "target" node
        H : iterable of ints
            a subset of the neighbors of y which are adjacent to x
        A : np.array
            the current adjacency matrix

        Returns
        -------
        new_A : np.array
            the adjacency matrix resulting from applying the operator

        """
        H = set(H)
        # Check inputs
        if A[x, y] == 0:
            raise ValueError(
                "There is no (un)directed edge from x=%d to y=%d" % (x, y))
        # neighbors of y which are adjacent to x
        na_yx = utils.na(y, x, A)
        if not H <= na_yx:
            raise ValueError(
                "The given set H is not valid, H=%s is not a subset of NA_yx=%s" % (
                    H, na_yx))
        # Apply operator
        new_A = A.copy()
        # delete the edge between x and y
        new_A[x, y], new_A[y, x] = 0, 0
        # orient the undirected edges between y and H towards H
        new_A[list(H), y] = 0
        # orient any undirected edges between x and H towards H
        n_x = utils.neighbors(x, A)
        new_A[list(H & n_x), x] = 0
        return new_A

    def score_valid_delete_operators(self, x, y, A, cache, debug=0):
        """Generate and score all valid delete(x,y,H) operators involving the edge
        x -> y or x - y, and all possible subsets H of neighbors of y which
        are adjacent to x.

        Parameters
        ----------
        x : int
            the "origin" node (i.e. x -> y or x - y)
        y : int
            the "target" node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        # Check inputs
        if A[x, y] == 0:
            raise ValueError(
                "There is no (un)directed edge from x=%d to y=%d" % (x, y))
        # One-hot encode all subsets of H0, plus one column to mark if
        # they have already passed the validity condition
        na_yx = utils.na(y, x, A)
        H0 = sorted(na_yx)
        p = len(A)
        if len(H0) == 0:
            subsets = np.zeros((1, (p + 1)), dtype=bool)
        else:
            subsets = np.zeros((2 ** len(H0), (p + 1)), dtype=bool)
            subsets[:, H0] = utils.cartesian([np.array([False, True])] * len(H0),
                                             dtype=bool)
        valid_operators = []
        print("    delete(%d,%d) H0=" % (x, y), set(H0)) if debug > 1 else None
        while len(subsets) > 0:
            print("      len(subsets)=%d, len(valid_operators)=%d" %
                  (len(subsets), len(valid_operators))) if debug > 1 else None
            # Access the next subset
            H = np.where(subsets[0, :-1])[0]
            cond_1 = subsets[0, -1]
            subsets = subsets[1:]
            # Check if the validity condition holds for H, i.e. that
            # NA_yx \ H is a clique.
            # If it has not been tested previously for a subset of H,
            # check it now
            if not cond_1 and utils.is_clique(na_yx - set(H), A):
                cond_1 = True
                # For all supersets H' of H, the validity condition will also hold
                supersets = subsets[:, H].all(axis=1)
                subsets[supersets, -1] = True
            # If the validity condition holds, apply operator and compute its score
            print("      delete(%d,%d,%s)" % (x, y, H), "na_yx - H = ",
                  na_yx - set(H), "validity:", cond_1) if debug > 1 else None
            if cond_1:
                # Apply operator
                new_A = self.delete_operator(x, y, H, A)
                # Compute the change in score
                aux = (na_yx - set(H)) | utils.pa(y, A) | {x}
                # print(x,y,H,"na_yx:",na_yx,"old:",aux,"new:", aux - {x})
                old_score = cache.local_score(y, aux)
                new_score = cache.local_score(y, aux - {x})
                print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" %
                      (y, aux - {x}, new_score, y, aux, old_score)) if debug > 1 else None
                # Add to the list of valid operators
                valid_operators.append((new_score - old_score, new_A, x, y, H))
                print("    delete(%d,%d,%s) -> %0.16f" %
                      (x, y, H, new_score - old_score)) if debug else None
        # Return all the valid operators
        return valid_operators

    def turn(self, x, y, C, A):
        # --------------------------------------------------------------------
        # Turn operator
        #    1. definition in function turn
        #    2. enumeration logic (to enumerate and score only valid
        #    operators) function in score_valid_insert_operators
        """
        Applies the turning operator: For an edge x - y or x <- y,
        1) orients the edge as x -> y
        2) for all c in C, orients the previously undirected edge c -> y

        Parameters
        ----------
        x : int
            the origin node (i.e. x -> y)
        y : int
            the target node
        C : iterable of ints
            a subset of the neighbors of y
        A : np.array
            the current adjacency matrix

        Returns
        -------
        new_A : np.array
            the adjacency matrix resulting from applying the operator

        """
        # Check inputs
        if A[x, y] != 0 and A[y, x] == 0:
            raise ValueError("The edge %d -> %d is already exists" % (x, y))
        if A[x, y] == 0 and A[y, x] == 0:
            raise ValueError("x=%d and y=%d are not connected" % (x, y))
        if not C <= utils.neighbors(y, A):
            raise ValueError(
                "Not all nodes in C=%s are neighbors of y=%d" % (C, y))
        if len({x, y} & C) > 0:
            raise ValueError("C should not contain x or y")
        # Apply operator
        new_A = A.copy()
        # Turn edge x -> y
        new_A[y, x] = 0
        new_A[x, y] = 1
        # Orient edges c -> y for c in C
        new_A[y, list(C)] = 0
        return new_A

    def score_valid_turn_operators(self, x, y, A, cache, debug=0):
        """Generate and score all valid turn(x,y,C) operators that can be
        applied to the edge x <- y or x - y, iterating through the valid
        subsets C of neighbors of y.

        Parameters
        ----------
        x : int
            the origin node (i.e. orient x -> y)
        y : int
            the target node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : int
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        if A[x, y] != 0 and A[y, x] == 0:
            raise ValueError(f"The edge {x} -> {y} already exists")
        if A[x, y] == 0 and A[y, x] == 0:
            raise ValueError(f"x={x} and y={y} are not connected")
        # Different validation/scoring logic when the edge to be turned is
        # essential (x <- x) or not (x - y)
        if A[x, y] != 0 and A[y, x] != 0:
            return self.score_valid_turn_operators_undir(x, y, A, cache, debug=debug)
        return self.score_valid_turn_operators_dir(x, y, A, cache, debug=debug)

    def score_valid_turn_operators_dir(self, x, y, A, cache, debug=0):
        """Logic for finding and scoring the valid turn operators that can be
        applied to the edge x <- y.

        Parameters
        ----------
        x : int
            the origin node (i.e. x -> y)
        y : int
            the target node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : bool or string
            if debug traces should be printed (True/False). If a non-empty
            string is passed, traces are printed with the given string as
            prefix (useful for indenting the prints from a calling
            function)

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        # One-hot encode all subsets of T0, plus one extra column to mark
        # if they pass validity condition 2 (see below). The set C passed
        # to the turn operator will be C = NAyx U T.
        p = len(A)
        T0 = sorted(utils.neighbors(y, A) - utils.adj(x, A))
        if len(T0) == 0:
            subsets = np.zeros((1, p + 1), dtype=bool)
        else:
            subsets = np.zeros((2 ** len(T0), p + 1), dtype=bool)
            subsets[:, T0] = utils.cartesian([np.array([False, True])] * len(T0),
                                             dtype=bool)
        valid_operators = []
        print("    turn(%d,%d) T0=" % (x, y), set(T0)) if debug > 1 else None
        while len(subsets) > 0:
            print("      len(subsets)=%d, len(valid_operators)=%d" %
                  (len(subsets), len(valid_operators))) if debug > 1 else None
            # Access the next subset
            T = np.where(subsets[0, :-1])[0]
            passed_cond_2 = subsets[0, -1]
            subsets = subsets[1:]  # update the list of remaining subsets
            # Check that the validity conditions hold for T
            C = utils.na(y, x, A) | set(T)
            # Condition 1: Test that C = NA_yx U T is a clique
            cond_1 = utils.is_clique(C, A)
            if not cond_1:
                # Remove from consideration all other sets T' which
                # contain T, as the clique condition will also not hold
                supersets = subsets[:, T].all(axis=1)
                subsets = utils.delete(subsets, supersets, axis=0)
            # Condition 2: Test that all semi-directed paths from y to x contain a
            # member from C U neighbors(x)
            if passed_cond_2:
                # If a subset of T satisfied condition 2, so does T
                cond_2 = True
            else:
                # otherwise, check condition 2
                cond_2 = True
                for path in utils.semi_directed_paths(y, x, A):
                    if path == [y, x]:
                        pass
                    elif len((C | utils.neighbors(x, A)) & set(path)) == 0:
                        cond_2 = False
                        break
                if cond_2:
                    # If condition 2 holds for C U neighbors(x), that is,
                    # for C = NAyx U T U neighbors(x), then it holds for
                    # all supersets of T
                    supersets = subsets[:, T].all(axis=1)
                    subsets[supersets, -1] = True
            # If both conditions hold, apply operator and compute its score
            print("      turn(%d,%d,%s)" % (x, y, C), "na_yx =", utils.na(y, x, A),
                  "T =", T, "validity:", cond_1, cond_2) if debug > 1 else None
            if cond_1 and cond_2:
                # Apply operator
                new_A = self.turn(x, y, C, A)
                # Compute the change in score
                new_score = cache.local_score(y, utils.pa(
                    y, A) | C | {x}) + cache.local_score(x, utils.pa(x, A) - {y})
                old_score = cache.local_score(y, utils.pa(y, A) | C) + \
                    cache.local_score(x, utils.pa(x, A))
                print("        new score = %0.6f, old score = %0.6f, y=%d, C=%s" %
                      (new_score, old_score, y, C)) if debug > 1 else None
                # Add to the list of valid operators
                valid_operators.append((new_score - old_score, new_A, x, y, C))
                print("    turn(%d,%d,%s) -> %0.16f" %
                      (x, y, C, new_score - old_score)) if debug else None
        # Return all the valid operators
        return valid_operators

    def score_valid_turn_operators_undir(self, x, y, A, cache, debug=0):
        """Logic for finding and scoring the valid turn operators that can be
        applied to the edge x - y.

        Parameters
        ----------
        x : int
            the origin node (i.e. x -> y)
        y : int
            the target node
        A : np.array
            the current adjacency matrix
        cache : instance of ges.scores.DecomposableScore
            the score cache to compute the score of the
            operators that are valid
        debug : bool or string
            if debug traces should be printed (True/False). If a non-empty
            string is passed, traces are printed with the given string as
            prefix (useful for indenting the prints from a calling
            function)

        Returns
        -------
        valid_operators : list of tuples
            a list of tubles, each containing a valid operator, its score
            and the resulting connectivity matrix

        """
        # Proposition 31, condition (ii) in GIES paper (Hauser & Bühlmann
        # 2012) is violated if:
        #   1. all neighbors of y are adjacent to x, or
        #   2. y has no neighbors (besides u)
        # then there are no valid operators.
        non_adjacents = list(utils.neighbors(y, A) - utils.adj(x, A) - {x})
        if len(non_adjacents) == 0:
            print("    turn(%d,%d) : ne(y) \\ adj(x) = Ø => stopping" % (
                x, y)) if debug > 1 else None
            return []
        # Otherwise, construct all the possible subsets which will satisfy
        # condition (ii), i.e. all subsets of neighbors of y with at least
        # one which is not adjacent to x
        p = len(A)
        C0 = sorted(utils.neighbors(y, A) - {x})
        subsets = np.zeros((2 ** len(C0), p + 1), dtype=bool)
        subsets[:, C0] = utils.cartesian(
            [np.array([False, True])] * len(C0), dtype=bool)
        # Remove all subsets which do not contain at least one non-adjacent node to x
        to_remove = (subsets[:, non_adjacents] == False).all(axis=1)
        subsets = utils.delete(subsets, to_remove, axis=0)
        # With condition (ii) guaranteed, we now check conditions (i,iii)
        # for each subset
        valid_operators = []
        print("    turn(%d,%d) C0=" % (x, y), set(C0)) if debug > 1 else None
        while len(subsets) > 0:
            print("      len(subsets)=%d, len(valid_operators)=%d" %
                  (len(subsets), len(valid_operators))) if debug > 1 else None
            # Access the next subset
            C = set(np.where(subsets[0, :])[0])
            subsets = subsets[1:]
            # Condition (i): C is a clique in the subgraph induced by the
            # chain component of y. Because C is composed of neighbors of
            # y, this is equivalent to C being a clique in A. NOTE: This
            # is also how it is described in Alg. 5 of the paper
            cond_1 = utils.is_clique(C, A)
            if not cond_1:
                # Remove from consideration all other sets C' which
                # contain C, as the clique condition will also not hold
                supersets = subsets[:, list(C)].all(axis=1)
                subsets = utils.delete(subsets, supersets, axis=0)
                continue
            # Condition (iii): Note that condition (iii) from proposition
            # 31 appears to be wrong in the GIES paper; instead we use the
            # definition of condition (iii) from Alg. 5 of the paper:
            # Let na_yx (N in the GIES paper) be the neighbors of Y which
            # are adjacent to X. Then, {x,y} must separate C and na_yx \ C
            # in the subgraph induced by the chain component of y,
            # i.e. all the simple paths from one set to the other contain
            # a node in {x,y}.
            subgraph = utils.induced_subgraph(utils.chain_component(y, A), A)
            na_yx = utils.na(y, x, A)
            if not utils.separates({x, y}, C, na_yx - C, subgraph):
                continue
            # At this point C passes both conditions
            #   Apply operator
            new_A = self.turn(x, y, C, A)
            #   Compute the change in score
            new_score = cache.local_score(y, utils.pa(
                y, A) | C | {x}) + cache.local_score(x, utils.pa(x, A) | (C & na_yx))
            old_score = cache.local_score(y, utils.pa(y, A) | C) + \
                cache.local_score(x, utils.pa(x, A) | (C & na_yx) | {y})
            print("        new score = %0.6f, old score = %0.6f, y=%d, C=%s" %
                  (new_score, old_score, y, C)) if debug > 1 else None
            #   Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, C))
            print("    turn(%d,%d,%s) -> %0.16f" % (
                x, y, C, new_score - old_score)) if debug else None
        # Return all valid operators
        return valid_operators


def main(dataset_name,
         input_path="/Users/renero/phd/data/sachs/",
         output_path="/Users/renero/phd/output/RC4/sachs/compared",
         save=False):

    ref_graph = graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    ges = GES(dataset_name, phases=["forward", "backward", "turning"], debug=0)
    ges.fit_predict(train=data, test=None, ref_graph=ref_graph)

    for edge in ges.dag.edges():
        print(edge)
    print(ges.metrics)

    # if save:
    #     where_to = utils.save_experiment(rex.name, output_path, rex)
    #     print(f"Saved '{rex.name}' to '{where_to}'")


if __name__ == "__main__":
    main("sachs_long")
