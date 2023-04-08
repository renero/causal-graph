# Structural Intervention Distance (SID) metric
#
# Based on original R implementation by Jonas Peters
# https://rdrr.io/cran/SID/f/
#

import time
from itertools import product
from typing import List

import igraph
import networkx as nx
import numpy as np
from causallearn.utils.DAG2CPDAG import dag2cpdag
from scipy.sparse import csr_matrix, diags, eye


def allDagsIntern(gm, a, row_names, tmp=None):
    if tmp is None:
        tmp = []

    if np.any(((a + a.T) == 1)):
        raise ValueError(
            'The matrix is not entirely undirected. This should not happen!')

    if a.sum() == 0:
        tmp2 = gm.flatten('F') if not len(tmp) else np.vstack([tmp, gm.flatten('F')])
        if all(np.logical_not(np.array([np.array_equal(x, y) for x in tmp2 for y in tmp2]).reshape(tmp2.shape[0], -1).sum(axis=1) > 1)):
            tmp = tmp2
    else:
        sinks = np.where(a.sum(axis=0) > 0)[1]
        for x in sinks:
            gm2 = gm.copy()
            a2 = None
            row_names2 = None

            Adj = (a == 1)
            Adjx = Adj[x, :]
            if np.any(Adjx):
                un = np.where(Adjx)[1]
                pp = len(un)
                Adj2 = np.matrix(Adj[un, :][:, un])
                np.fill_diagonal(Adj2, True)
            else:
                Adj2 = True

            if np.all(Adj2):
                if np.any(Adjx):
                    un = row_names[np.where(Adjx)[1]]
                    pp = len(un)
                    gm2[un, row_names[x]] = np.ones(pp)
                    gm2[row_names[x], un] = np.zeros(pp)

                a2 = np.delete(np.delete(a, x, axis=0), x, axis=1)
                row_names2 = np.delete(row_names, x)

                # print("~"*100)
                # print("gm2")
                # for ridx in range(gm2.shape[0]):
                #     print(str(gm2[ridx, :])[2:-2])
                # print("a2")
                # for ridx in range(a2.shape[0]):
                #     print(str(a2[ridx, :])[2:-2])
                
                tmp = allDagsIntern(gm2, a2, row_names2, tmp)

    return tmp


def allDagsJonas(adj: np.matrix, row_names: List[str]) -> List[List[str]]:
    # Input: adj. mat of a DAG with row.names, containing the undirected component that
    #        should be extended
    # !!!! the function can probably be faster if we use partial orderings

    a = adj[row_names, :][:, row_names]
    if np.any(((a + a.T) == 1)):
        # if any((a + a.T) == 1):
        # warning("The matrix is not entirely undirected.")
        return -1
    return allDagsIntern(adj, a, row_names, None)


def computePathMatrix(G, spars=False):
    # this function takes an adjacency matrix G from a DAG and computes a path matrix for which
    # entry(i,j) being one means that there is a directed path from i to j
    # the diagonal will also be one
    p = G.shape[1]

    if (p > 3000) and (not spars):
        print("Warning: Maybe you should use the sparse version by using spars=True to increase speed")

    if spars:
        G = csr_matrix(G)
        PathMatrix = diags(np.ones(p), 0) + G
    else:
        PathMatrix = np.eye(p) + G

    k = int(np.ceil(np.log(p) / np.log(2)))
    for i in range(k):
        PathMatrix = PathMatrix.dot(PathMatrix)

    PathMatrix = PathMatrix > 0

    return PathMatrix


def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    # The only difference to the function computePathMatrix is that this function changes
    # the graph by removing all edges that leave condSet.
    # If condSet is empty, it just returns PathMatrix1.

    p = G.shape[1]
    if len(condSet) > 0:
        G[condSet, :] = np.zeros((len(condSet), p))

        if spars:
            G = csr_matrix(G)
            PathMatrix2 = eye(p) + G
        else:
            PathMatrix2 = np.eye(p) + G

        k = int(np.ceil(np.log(p) / np.log(2)))
        for i in range(k):
            PathMatrix2 = PathMatrix2.dot(PathMatrix2)
        PathMatrix2 = PathMatrix2 > 0
    else:
        PathMatrix2 = PathMatrix1
    return PathMatrix2


def compute_caus_order(G):
    p = G.shape[1]
    remaining = list(range(1, p + 1))
    caus_order = [None] * p
    for i in range(p - 1):
        root = min(index for index, value in enumerate(
            G.sum(axis=0)) if value == 0)
        caus_order[i] = remaining[root]
        remaining.pop(root)
        G = np.delete(np.delete(G, root, axis=0), root, axis=1)
    caus_order[p - 1] = remaining[0]
    return caus_order


def dag2cpdagAdj(Adj):
    if np.sum(Adj) == 0:
        return Adj

    cO = compute_caus_order(Adj)
    d = nx.from_numpy_matrix(Adj[np.ix_(cO, cO)])
    cpd = dag2cpdag(d)
    res = np.empty((Adj.shape[0], Adj.shape[1]), dtype=float)
    res[np.ix_(cO, cO)] = nx.to_numpy_matrix(cpd)
    result = res

    return result


def dSepAdji(AdjMat, i, condSet, PathMatrix=None, PathMatrix2=None, spars=None):
    print("\n" + "*"*40)
    print("* i:", i, "; condSet:", condSet, "; spars:", spars, "*")
    print("*"*40 + "\n")

    if PathMatrix is None:
        PathMatrix = computePathMatrix(AdjMat.copy())
    if PathMatrix2 is None:
        PathMatrix2 = np.full((p, p), np.nan)
    if spars is None:
        spars = (p > 99)

    timeComputePM2 = 0
    timeComputePM = 0
    if np.isnan(PathMatrix2.sum()):
        ptm = time.process_time()
        PathMatrix2 = computePathMatrix2(AdjMat.copy(), condSet, PathMatrix)
        timeComputePM2 += time.process_time() - ptm

    if len(condSet) == 0:
        AncOfCondSet = []
    elif len(condSet) == 1:
        AncOfCondSet = np.where(PathMatrix[:, condSet] > 0)[0]
    else:
        AncOfCondSet = np.where(np.sum(PathMatrix[:, condSet], axis=1) > 0)[0]

    p = AdjMat.shape[1]
    reachabilityMatrix = np.zeros((2 * p, 2 * p)).astype(int)
    reachableOnNonCausalPathLater = np.zeros((2, 2)).astype(int)

    reachableNodes = np.zeros(2 * p).astype(int)
    reachableOnNonCausalPath = np.zeros(2 * p).astype(int)
    alreadyChecked = np.zeros(p).astype(int)
    k = 1
    toCheck = [0, 0]

    reachableCh = np.where(AdjMat[i, :] == 1)[1]
    print("    +++ reachableCh:", reachableCh)
    if len(reachableCh) > 0:
        toCheck.extend(reachableCh)
        reachableNodes[reachableCh] = [1] * len(reachableCh)
        AdjMat[i, reachableCh] = [0] * len(reachableCh)

    reachablePa = np.where(AdjMat[:, i] == 1)[0]
    print("    +++ reachablePa:", reachablePa)
    if len(reachablePa) > 0:
        toCheck.extend(reachablePa)
        reachableNodes[reachablePa + p] = [1] * len(reachablePa)
        reachableOnNonCausalPath[reachablePa + p] = [1] * len(reachablePa)
        AdjMat[reachablePa, i] = [0] * len(reachablePa)
    print("    >>> P0: reachableOnNonCausalPath:", reachableOnNonCausalPath)

    while k < len(toCheck)-1:
        k += 1
        a1 = toCheck[k]
        print("[ k:",k,"; a1:",a1,"; toCheck:",toCheck, "]")
        # Pretty print the reachability matrix
        print("Reachability Matrix:")
        for ii in range(2 * p):
            for jj in range(2 * p):
                print(reachabilityMatrix[ii, jj], " ", end='')
            print()
        if alreadyChecked[a1] == 0:
            currentNode = a1
            alreadyChecked[a1] = 1

            Pa = np.where(AdjMat[:, currentNode] == 1)[0]
            Pa1 = np.setdiff1d(Pa, condSet)
            # reachabilityMatrix[Pa1, currentNode] = 1
            # reachabilityMatrix[Pa1 + p, currentNode] = 1
            reachabilityMatrix[Pa1, currentNode] = [1] * len(Pa1)
            reachabilityMatrix[Pa1 + p, currentNode] = [1] * len(Pa1)
            if len(Pa1):
                print("L0 -> CurNode:",currentNode, "; Pa1:",Pa1,"; len(Pa1):",len(Pa1))

            if np.sum(np.isin(AncOfCondSet, currentNode)) > 0:
                reachabilityMatrix[currentNode, Pa + p] = 1
                if PathMatrix2[i, currentNode] > 0:
                    reachableOnNonCausalPathLater = np.vstack(
                        (reachableOnNonCausalPathLater, np.column_stack(
                        (np.repeat(currentNode, len(Pa)), Pa))))
                newtoCheck = Pa[np.where(alreadyChecked[Pa] == 0)[0]]
                toCheck.extend(newtoCheck)

            if np.sum(np.isin(condSet, currentNode)) == 0:
                reachabilityMatrix[currentNode + p, Pa + p] = [1] * len(Pa)
                if len(Pa):
                    print("L1 -> CurNode:",currentNode, "; Pa:",Pa,"; len(Pa):",len(Pa))
                newtoCheck = Pa[np.where(alreadyChecked[Pa] == 0)[0]]
                toCheck.extend(newtoCheck)

            Ch = np.where(AdjMat[currentNode, :] == 1)[1]
            Ch1 = np.setdiff1d(Ch, condSet)
            if len(Ch1):
                reachabilityMatrix[Ch1 + p, currentNode + p] = [1] * len(Ch1)
                print("L2 -> CurNode:",currentNode, "; Ch1:", Ch1, "; len(Ch1):",len(Ch1))

            Ch2 = np.intersect1d(Ch, AncOfCondSet)
            if len(Ch2):
                reachabilityMatrix[Ch2, currentNode + p] = [1] * len(Ch2)
                print("L3 -> CurNode:",currentNode, "; Ch2:", Ch2, "; len(Ch2):",len(Ch2))
            Ch2b = np.intersect1d(Ch2, np.where(PathMatrix2[i, :] > 0)[1])
            if len(Ch2b):
                reachableOnNonCausalPathLater = np.vstack(
                    (reachableOnNonCausalPathLater, np.column_stack(
                    (Ch2b, np.repeat(currentNode, len(Ch2b))))))

            if np.sum(np.isin(condSet, currentNode)) == 0:
                reachabilityMatrix[currentNode, Ch] = [1] * len(Ch)
                reachabilityMatrix[currentNode + p, Ch] = [1] * len(Ch)
                if len(Ch):
                    print("L4 -> CurNode:",currentNode, "; Ch:",Ch,"; len(Ch):",len(Ch))
                newtoCheck = Ch[np.where(alreadyChecked[Ch] == 0)[0]]
                toCheck.extend(newtoCheck)
        print("-"*60)

    reachabilityMatrix = computePathMatrix(reachabilityMatrix, spars=spars)
    # reachabilityMatrix = reachabilityMatrix.toarray()

    ttt2 = np.where(reachableNodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachabilityMatrix[ttt2, :] > 0)[0]
    else:
        tt2 = np.where(np.sum(reachabilityMatrix[ttt2, :], axis=0) > 0)[0]
    reachableNodes[tt2] = 1

    ttt = np.where(reachableOnNonCausalPath == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachabilityMatrix[ttt, :] > 0)[1]
    else:
        tt = np.where(np.sum(reachabilityMatrix[ttt, :], axis=0) > 0)[0]
    reachableOnNonCausalPath[tt] = 1
    print("    >>> P1: reachableOnNonCausalPath:", reachableOnNonCausalPath)


    if reachableOnNonCausalPathLater.shape[0] > 2:
        for kk in range(2, reachableOnNonCausalPathLater.shape[0]):
            ReachableThrough = reachableOnNonCausalPathLater[kk, 0]
            newReachable = reachableOnNonCausalPathLater[kk, 1]
            reachableOnNonCausalPath[newReachable + p] = 1
            print("    >>> P2: reachableOnNonCausalPath:", reachableOnNonCausalPath)

            reachabilityMatrix[newReachable, ReachableThrough] = 0
            reachabilityMatrix[newReachable, ReachableThrough + p] = 0
            reachabilityMatrix[newReachable + p, ReachableThrough] = 0
            reachabilityMatrix[newReachable + p, ReachableThrough + p] = 0

        ttt = np.where(reachableOnNonCausalPath == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachabilityMatrix[ttt, :] > 0)[0]
        else:
            tt = np.where(np.sum(reachabilityMatrix[ttt, :], axis=0) > 0)[0]
        reachableOnNonCausalPath[tt] = 1
        print("    >>> P3: reachableOnNonCausalPath:", reachableOnNonCausalPath)

    result = {}
    result['timeComputePM'] = timeComputePM
    result['timeComputePM2'] = timeComputePM2
    result['reachableJ'] = np.sum(np.column_stack(
        (reachableNodes[:p], reachableNodes[p:(2 * p)])), axis=1) > 0
    result['reachableOnNonCausalPath'] = np.sum(np.column_stack(
        (reachableOnNonCausalPath[:p], reachableOnNonCausalPath[p:(2 * p)])), axis=1) > 0
    print("    >>> P4: reachableOnNonCausalPath:", reachableOnNonCausalPath)

    return result


def unique_rows(m):
    mm = np.ascontiguousarray(m)
    result = []
    uniques = set()
    for row_index in range(0, mm.shape[0]):
        row = mm[row_index, :]
        tup = tuple(row)
        if tup not in uniques:
            uniques.add(tup)
            result.append(row_index)
    return result


def SID(trueGraph: np.ndarray, estGraph: np.ndarray, output: bool = False, spars: bool = False) -> float:
    estGraph = np.array(estGraph)
    trueGraph = np.array(trueGraph)
    p = trueGraph.shape[1]
    estGraph = np.matrix(estGraph)
    trueGraph = np.matrix(trueGraph)
    incorrectInt = np.zeros((p, p))
    correctInt = np.zeros((p, p))
    minimumTotal = 0
    maximumTotal = 0
    numChecks = 0

    PathMatrix = computePathMatrix(trueGraph.copy(), spars)
    PathMatrix = np.matrix(PathMatrix)

    Gp_undir = np.multiply(estGraph, estGraph.T)
    # Build an undirected graph from Gp_undir adjacency matrix
    gp_undir = nx.from_numpy_array(Gp_undir)
    # gp_undir = np.array(Gp_undir)
    conn_comp = list(nx.connected_components(gp_undir))
    numConnComp = len(conn_comp)
    GpIsEssentialGraph = True
    for ll in range(numConnComp):
        conn_comp[ll] = np.array(list(conn_comp[ll]), dtype=int)
        if len(conn_comp[ll]) > 1:
            # undir_adjacency = Gp_undir[np.ix_(conn_comp[ll], conn_comp[ll])]
            # undir_graph = nx.from_numpy_array(undir_adjacency, create_using=nx.Graph)
            # chordal = nx.is_chordal(undir_graph)
            chordal = igraph._igraph.GraphBase.is_chordal(
                igraph.Graph.Weighted_Adjacency(
                    Gp_undir[np.ix_(conn_comp[ll], conn_comp[ll])].tolist(),
                    mode="undirected"
                )
            )
            if not chordal:
                GpIsEssentialGraph = False
            if len(conn_comp[ll]) > 8:
                GpIsEssentialGraph = False

    for ll in range(numConnComp):
        # if len(conn_comp[ll]) > 0:
        if len(conn_comp[ll]) <= 0:
            continue
        if GpIsEssentialGraph:
            if len(conn_comp[ll]) > 1:
                mmm = allDagsJonas(estGraph, conn_comp[ll])
            else:
                # mmm = np.matrix(estGraph).reshape(1, p**2)
                mmm = estGraph.flatten(order='F')
            if np.sum(mmm == -1) == 1:
                GpIsEssentialGraph = False
                # mmm = np.matrix(estGraph).reshape(1, p**2)
                mmm = estGraph.flatten(order='F')
            newInd = np.arange(
                1, p**3, p) - np.repeat(np.arange(0, (p-1)*((p**2)-1)+1, (p**2)-1), p)
            # Fix that Python starts counting from 0 instead 1, in R.
            newInd = newInd - 1
            dimM = mmm.shape
            mmm = np.matrix(mmm[:, newInd]).reshape(dimM)

            if mmm is None:
                GpIsEssentialGraph = False
            else:
                incorrectSum = np.zeros(mmm.shape[0])
        
        for i in conn_comp[ll]:
            paG = np.where(trueGraph[:, i] == 1)[0]
            print(">>> set paG to:", paG)
            certainpaGp = np.where(
                np.multiply(estGraph[:, i].T, (np.ones(p) - estGraph[i, :])) == 1)[1]
            print(">>> set certainpaGp to:", certainpaGp)
            possiblepaGp = np.where(
                np.multiply(estGraph[:, i].T, estGraph[i, :]) == 1)[1]
            print(">>> set possiblepaGp to:", possiblepaGp)
            if not GpIsEssentialGraph:
                maxcount = 2**len(possiblepaGp)
                uniqueRows = np.arange(1, maxcount+1)
                mmm = np.tile(estGraph.T.flatten(), (maxcount, 1))
                mmm[:, i + (possiblepaGp - 1) * p] = np.array(
                    list(product([0, 1], repeat=len(possiblepaGp))))
                incorrectSum = np.zeros(maxcount)
            else:
                if mmm.shape[0] > 1:
                    allParentsOfI = np.arange(i, ((p-1)*p+i)+1, p)
                    uniqueRows = unique_rows(mmm[:, allParentsOfI])
                    maxcount = len(uniqueRows)
                else:
                    maxcount = 1
                    uniqueRows = [0]

            count = 1
            while count <= maxcount:
                if maxcount == 1:
                    paGp = certainpaGp
                else:
                    Gpnew = np.matrix(
                        mmm[uniqueRows[count-1], :]).reshape(p, p)
                    paGp = np.where(Gpnew[:, i] == 1)[0]
                    if output:
                        print(
                            f"{i} has {len(paGp)} parents in expansion nr. {uniqueRows[count-1]} of Gp:")
                        print(paGp)

                PathMatrix2 = computePathMatrix2(
                    trueGraph.copy(), paGp, PathMatrix, spars)

                checkAlldSep = dSepAdji(
                    trueGraph.copy(), i, paGp, PathMatrix, PathMatrix2, spars=spars)
                
                numChecks += 1
                reachableWOutCausalPath = checkAlldSep["reachableOnNonCausalPath"]

                for j in range(p):
                    if i == j:
                        continue
                    finished = False
                    ijGNull = False
                    ijGpNull = False

                    if PathMatrix[i, j] == 0:
                        ijGNull = True

                    if (sum(paGp == j) == 1):
                        ijGpNull = True

                    if ijGpNull and ijGNull:
                        finished = True
                        correctInt[i, j] = 1

                    print(f">>> i={i}, j={j}, ijGNull={ijGNull}, ijGpNull={ijGpNull}, finished={finished}")

                    if ijGpNull and not ijGNull:
                        incorrectInt[i, j] = 1
                        incorrectSum[uniqueRows[count-1]] += 1
                        allOthers = np.setdiff1d(np.arange(0, mmm.shape[0]), uniqueRows)
                        if len(allOthers) > 1:
                            indInAllOthers = np.where(np.sum(~np.logical_xor(
                                mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI]), axis=1) == p)[0]
                            if len(indInAllOthers) > 0:
                                incorrectSum[allOthers[indInAllOthers] -
                                                1] += np.ones(len(indInAllOthers))
                        if len(allOthers) == 1:
                            indInAllOthers = np.where(np.sum(~np.logical_xor(
                                mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI])) == p)[0]
                            if len(indInAllOthers) > 0:
                                incorrectSum[allOthers[indInAllOthers] -
                                                1] += np.ones(len(indInAllOthers))
                        finished = True

                    if not finished and set(paG) == set(paGp):
                        finished = True
                        correctInt[i, j] = 1

                    if not finished:
                        if PathMatrix[i, j] > 0:
                            chiCausPath = np.where(
                                trueGraph[i, :] & PathMatrix[:, j].T)[1]
                            if np.sum(PathMatrix[chiCausPath, paGp]) > 0:
                                incorrectInt[i, j] = 1
                                incorrectSum[uniqueRows[count-1]] += 1
                                allOthers = np.setdiff1d(
                                    np.arange(0, mmm.shape[0]), uniqueRows)
                                if len(allOthers) > 1:
                                    indInAllOthers = np.where(np.sum(~np.logical_xor(
                                        mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI]), axis=1) == p)[0]
                                    if len(indInAllOthers) > 0:
                                        incorrectSum[allOthers[indInAllOthers] -
                                                        1] += np.ones(len(indInAllOthers))
                                if len(allOthers) == 1:
                                    indInAllOthers = np.where(np.sum(~np.logical_xor(
                                        mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI])) == p)[0]
                                    if len(indInAllOthers) > 0:
                                        incorrectSum[allOthers[indInAllOthers] -
                                                        1] += np.ones(len(indInAllOthers))
                                finished = True

                        if not finished:
                            if reachableWOutCausalPath[j] == 1:
                                incorrectInt[i, j] = 1
                                incorrectSum[uniqueRows[count-1]] += 1
                                allOthers = np.setdiff1d(
                                    np.arange(0, mmm.shape[0]), uniqueRows)
                                if len(allOthers) > 1:
                                    indInAllOthers = np.where(np.sum(~np.logical_xor(
                                        mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI]), axis=1) == p)[0]
                                    if len(indInAllOthers) > 0:
                                        incorrectSum[allOthers[indInAllOthers] -
                                                        1] += np.ones(len(indInAllOthers))
                                if len(allOthers) == 1:
                                    indInAllOthers = np.where(np.sum(~np.logical_xor(
                                        mmm[uniqueRows[count-1], allParentsOfI], mmm[allOthers-1, allParentsOfI])) == p)[0]
                                    if len(indInAllOthers) > 0:
                                        incorrectSum[allOthers[indInAllOthers] -
                                                        1] += np.ones(len(indInAllOthers))
                            else:
                                correctInt[i, j] = 1
                count += 1
        if not GpIsEssentialGraph:
            minimumTotal += np.min(incorrectSum)
            maximumTotal += np.max(incorrectSum)
            incorrectSum = 0
        minimumTotal += np.min(incorrectSum)
        maximumTotal += np.max(incorrectSum)
        incorrectSum = 0

    ress = {}
    ress["sid"] = np.sum(incorrectInt)
    ress["sidUpperBound"] = maximumTotal
    ress["sidLowerBound"] = minimumTotal
    ress["incorrectMat"] = incorrectInt

    return ress


def hammingDist(G1, G2, allMistakesOne=True):
    # hammingDist(G1,G2)
    #
    # Computes Hamming Distance between DAGs G1 and G2 with SHD(->,<-) = 1 if allMistakesOne == TRUE
    #
    # INPUT:  G1, G2     adjacency graph containing only zeros and ones: (i,j)=1 means edge from X_i to X_j.
    #
    # OUTPUT: hammingDis Hamming Distance between G1 and G2
    if allMistakesOne:
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nrReversals = np.sum(Gtmp == 2) / 2
        nrInclDel = np.sum(Gtmp == 1) / 2
        hammingDis = nrReversals + nrInclDel
    else:
        hammingDis = np.sum(np.abs(G1 - G2))
        hammingDis = hammingDis - 0.5 * \
            np.sum(G1 * G1.T * (1 - G2) * (1 - G2).T +
                   G2 * G2.T * (1 - G1) * (1 - G1).T)

    return hammingDis


if __name__ == "__main__":
    G = np.array([[0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])

    H1 = np.array([[0, 1, 1, 1, 1],
                   [0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    H2 = np.array([[0, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    H1c = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]])

    # print("true DAG G:")
    # print(G)
    # print("==============")
    # print("estimated DAG H1:")
    # print(H1)
    # sid = SID(G, H1)
    # shd = hammingDist(G, H1)
    # print(f"\n\nSHD between G and H1: {shd}")
    # print(f"SID between G and H1: {sid['sid']}")
    # print("===========================\n\n")

    # print("estimated DAG H2:")
    # print(H2)
    # sid = SID(G, H2)
    # shd = hammingDist(G, H2)
    # print(f"SHD between G and H2: {shd}")
    # print(f"SID between G and H2: {sid['sid']}")
    # print("The matrix of incorrect interventional distributions is:")
    # print(sid['incorrectMat'])

    # input("The SID can also be applied to CPDAGs. Please press enter...")
    print("==============")
    print("estimated CPDAG H1c:")
    print(H1c)
    sid = SID(G, H1c)
    print(f"SID between G and H1:")
    print(
        f"lower bound: {sid['sidLowerBound']} upper bound: {sid['sidUpperBound']}")
