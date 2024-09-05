"""
- **Imports**: Imported necessary modules and functions. Assumed that
`computeScoreMat`, `updateScoreMat`, `pruning`, `selGamBoost`, and `selGam`
are defined in separate Python files in the same directory.
- **Function Definition**: Translated the R function `CAM` to Python.
- **Variable Initialization**: Initialized variables and handled default values.
- **Variable Selection**: Used `numpy` and `multiprocessing` for parallel processing.
- **Edge Inclusion**: Translated the logic for including edges and updating the
score matrix.
- **Pruning**: Translated the pruning step.
- **Output and Return**: Collected and printed the results.

Make sure the corresponding Python files (`computeScoreMat.py`, `updateScoreMat.py`,
`pruning.py`, `selGamBoost.py`, `selGam.py`) are present in the same directory and
contain the necessary functions.
"""
# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, E0401:import-error
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import sys
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from computeScoreMat import computeScoreMat
from pruning import pruning
from selGam import selGam
from selGamBoost import selGamBoost
from updateScoreMat import updateScoreMat

from causalgraph.metrics.compare_graphs import evaluate_graph
from causalgraph.common.utils import graph_from_adjacency


class CAM:
    def __init__(
            self,
            scoreName="SEMGAM",
            parsScore=None,
            numCores=1,
            maxNumParents=None,
            verbose=False,
            variableSel=False,
            variableSelMethod=selGamBoost,
            variableSelMethodPars=None,
            pruning=False,
            pruneMethod=selGam,
            pruneMethodPars=None,
            intervData=False,
            intervMat=None):

        self.scoreName = scoreName
        self.parsScore = parsScore
        self.numCores = numCores
        self.maxNumParents = maxNumParents
        self.verbose = verbose
        self.variableSel = variableSel
        self.variableSelMethod = variableSelMethod
        self.variableSelMethodPars = variableSelMethodPars
        self.pruning = pruning
        self.pruneMethod = pruneMethod
        self.pruneMethodPars = pruneMethodPars
        self.intervData = intervData
        self.intervMat = intervMat

        self.is_fitted_ = False
        self.metrics = None
        self.dag = None

        if parsScore is None:
            self.parsScore = {"numBasisFcts": 10}
        if variableSelMethodPars is None:
            self.variableSelMethodPars = {
                "atLeastThatMuchSelected": 0.02, "atMostThatManyNeighbors": 10}
        if pruneMethodPars is None:
            self.pruneMethodPars = {"cutOffPVal": 0.001, "numBasisFcts": 10}

        self.maxNumParents = maxNumParents

    def fit(self, X):
        """
        This method implements the entire CAM algorithm. Translated from the R code.

        Parameters
        ----------
        X : np.array
            Observational data

        Returns
        -------
        edgeList : list
            List of edges
        scoreVec : list
            List of scores
        """
        if self.verbose:
            print(f"number of cores: {self.numCores}")

        if self.maxNumParents is None:
            self.maxNumParents = min(X.shape[1] - 1, round(X.shape[0] / 20))

        timeCycle = 0
        timeUpdate = 0
        timeScoreMat = 0
        timeSel = 0
        timePrune = 0
        timeMax = 0

        scoreVec = []
        edgeList = []

        counterUpdate = 0
        p = X.shape[1]

        if self.variableSel:
            start_time = time.time()
            if self.intervData:
                X2 = X[np.sum(self.intervMat, axis=1) == 0, :]
                if self.verbose:
                    print("The preliminary neighbourhood selection is done with the " +
                          "observational data only.")
            else:
                X2 = X

            if self.numCores == 1:
                selMat = np.array(
                    [self.variableSelMethod(X2, self.variableSelMethodPars, self.verbose) for _ in range(p)])
            else:
                with Pool(self.numCores) as pool:
                    selMat = np.array(pool.starmap(self.variableSelMethod, [
                        (X2, self.variableSelMethodPars, self.verbose)
                        for _ in range(p)]))

            cou = sum(2 ** np.sum(selMat[:, jk]) for jk in range(p))
            if self.verbose:
                print(
                    f"Instead of p2^(p-1) -Sillander- {p * 2 ** (p - 1)} we have {cou}")
                print(
                    f"Greedy, on the other hand, is computing {np.sum(selMat)} entries.")

            timeSel += time.time() - start_time
        else:
            selMat = np.ones((p, p), dtype=bool)
            print('NO variable selection') if self.verbose else None

        if self.variableSel and self.verbose:
            if p < 30:
                print("This is the matrix of possible parents after the first step.")
                print(selMat)

        start_time = time.time()
        computeScoreMatTmp = computeScoreMat(
            X, score_name=self.scoreName, num_parents=1, num_cores=self.numCores,
            verbose=self.verbose, sel_mat=selMat, pars_score=self.parsScore, interv_mat=self.intervMat,
            interv_data=self.intervData)
        timeScoreMat += time.time() - start_time

        pathMatrix = np.eye(p, dtype=int)
        self.adj = np.zeros((p, p), dtype=int)
        scoreNodes = computeScoreMatTmp['scoreEmptyNodes']

        if self.verbose:
            print(f"scoreNodes: {scoreNodes}")
            print("Contents of computeScoreMat@CAM()")
            for i in range(computeScoreMatTmp["scoreMat"].shape[0]):
                for j in range(computeScoreMatTmp["scoreMat"].shape[1]):
                    if computeScoreMatTmp['scoreMat'][i, j] == -np.inf:
                        print(" -inf   ", end="")
                    else:
                        print(
                            f"{computeScoreMatTmp['scoreMat'][i, j]:+.4f} ", end="")
                print("")

            print("Pre-WHILE condition: ", np.sum(computeScoreMatTmp['scoreMat'] != -np.Inf))

        while np.sum(computeScoreMatTmp['scoreMat'] != -np.inf) > 0:
            start_time = time.time()
            ix_max = np.unravel_index(np.argmax(
                computeScoreMatTmp['scoreMat']), computeScoreMatTmp['scoreMat'].shape)

            timeMax += time.time() - start_time
            self.adj[ix_max] = 1
            scoreNodes[ix_max[1]] += computeScoreMatTmp['scoreMat'][ix_max]

            if self.verbose:
                print(f"ix_max: {ix_max}")
                print(f"\nIncluded edge (from, to) {ix_max}")

            computeScoreMatTmp['scoreMat'][ix_max] = -np.inf

            start_time = time.time()
            pathMatrix[ix_max[0], ix_max[1]] = 1
            DescOfNewChild = np.where(pathMatrix[ix_max[1], :] == 1)[0]
            AncOfNewParent = np.where(pathMatrix[:, ix_max[0]] == 1)[0]
            pathMatrix[np.ix_(AncOfNewParent, DescOfNewChild)] = 1
            computeScoreMatTmp['scoreMat'][pathMatrix.T == 1] = -np.inf
            computeScoreMatTmp['scoreMat'][ix_max[1], ix_max[0]] = -np.inf
            timeCycle += time.time() - start_time

            scoreVec.append(np.sum(scoreNodes))
            edgeList.append(ix_max)

            start_time = time.time()
            computeScoreMatTmp['scoreMat'] = updateScoreMat(
                computeScoreMatTmp['scoreMat'], X, score_name=self.scoreName, i=ix_max[0],
                j=ix_max[1], score_nodes=scoreNodes, adj=self.adj, num_cores=self.numCores,
                verbose=self.verbose, max_num_parents=self.maxNumParents, pars_score=self.parsScore,
                interv_mat=self.intervMat, interv_data=self.intervData)
            timeUpdate += time.time() - start_time

            counterUpdate += 1

        if self.verbose:
            print("\n--------------------------")
            print("Finished step 2 ----------")
            print("--------------------------")
        # Step 3

        if self.pruning:
            if self.intervData:
                X2 = X[np.sum(self.intervMat, axis=1) == 0, :]
                print("The preliminary neighbourhood selection is done with the " +
                      "observational data only.")
            else:
                X2 = X

            if self.verbose:
                print("\n Performing pruning ... \n")

            start_time = time.time()
            self.adj = pruning(X=X2, G=self.adj, prune_method=self.pruneMethod,
                          prune_method_pars=self.pruneMethodPars, verbose=self.verbose)
            timePrune += time.time() - start_time

        timeTotal = timeSel + timeScoreMat + timeCycle + timeUpdate + timeMax + timePrune
        if self.verbose:
            print("\nTiming:")
            print(f"  Time for variable selection: {timeSel:.2f}")
            print(f"  Time computing the initial scoreMat: {timeScoreMat:.2f}")
            print(f"  Time checking for cycles: {timeCycle:.2f}")
            print(
                f"  Time computing updates for the scoreMat: {timeUpdate:.2f}, "
                f"doing {counterUpdate} updates.")
            print(f"  Time for pruning: {timePrune:.2f}")
            print(f"  Time for finding maximum: {timeMax:.2f}")
            print(f"  Time in total: {timeTotal:.2f}")

        result = {
            "Adj": self.adj,
            "Score": np.sum(scoreNodes),
            "timesVec":
                [timeSel, timeScoreMat, timeCycle, timeUpdate,
                    timePrune, timeMax, timeTotal],
            "scoreVec": scoreVec,
            "edgeList": edgeList
        }

        self.is_fitted_ = True
        return self

    def predict(self, ref_graph: nx.DiGraph = None):
        if not self.is_fitted_:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                f"Call 'fit' with appropriate arguments before using this method.")

        self.dag = graph_from_adjacency(self.adj)
        if ref_graph is not None:
            self.metrics = evaluate_graph(
                ref_graph, self.dag, self.feature_names)

        return self.dag

    def fit_predict(self, X, ref_graph: nx.DiGraph = None):
        self.fit(X)
        self.predict(ref_graph)

        return self.dag


def main(dataset_name,
         input_path="/Users/renero/phd/data/RC3/",
         output_path="/Users/renero/phd/output/RC4/",
         scale_data: bool = False,
         tune_model: bool = False,
         model_type="nn", explainer="gradient",
         save=False):

    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    cam = CAM(pruning=True, verbose=False)
    cam.fit_predict(data.values)
    print(cam.dag)


if __name__ == "__main__":
    main("short")
