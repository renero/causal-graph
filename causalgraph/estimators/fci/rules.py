"""
This module contains the rules for edge orientation in the fci algorithm
"""
import networkx as nx


def rule1(pag, i, j, k):
    """
    rule 1 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k: str
            nodes to test for orientation rule
    """
    if pag.has_directed_edge(i, j) and pag.has_o(j, k, j) and not pag.has_edge(i, k):
        pag.fully_direct_edge(j, k)
        print(f"Orienting edge {j},{k} with rule 1")
        return True
    return False


def rule2(pag, i, j, k):
    """
    rule 2 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k: str
            nodes to test for orientation rule
    """
    chain1 = pag.has_fully_directed_edge(i, j) and pag.has_directed_edge(j, k)
    chain2 = pag.has_fully_directed_edge(j, k) and pag.has_directed_edge(i, j)
    if (chain1 or chain2) and pag.has_o(i, k, k):
        pag.direct_edge(i, k)
        # print("Orienting edge {},{} with rule 2".format(i, k))
        return True
    return False


def rule3(pag, i, j, k, l):
    """
    rule 3 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k,l: str
            nodes to test for orientation rule
    """
    chain1 = (pag.has_directed_edge(i, j)) and pag.has_directed_edge(k, j)
    chain2 = (pag.has_o(i, l, l)) and (pag.has_o(k, l, l))
    if chain1 and chain2 and not pag.has_edge(i, k) and pag.has_o(l, j, j):
        pag.direct_edge(l, j)
        # print("Orienting edge {},{} with rule 3".format(l, j))
        return True
    return False


def rule4(pag, i, j, k, node, sepSet):
    """
    rule 4 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k,node: str
            nodes to test for orientation rule
        sepSet: dict
    """
    paths = pag.findDiscPath(node, k, j)
    changes = False
    for path in (path for path in paths if i in path):
        if path.index(i) == len(path) - 3 and pag.has_o(j, k, j):
            if j in sepSet[(node, k)]:
                pag.fully_direct_edge(j, k)
                # print("Orienting edge {},{} with rule 4".format(j, k))
                changes = True
            else:
                pag.direct_edge(i, j)
                pag.direct_edge(j, k)
                pag.direct_edge(j, i)
                pag.direct_edge(k, j)
                # print("Orienting edges {},{}, {},{} with rule 4".format(i, j, j, k))
                changes = True
    return changes


def rule5(pag, i, j, k, node):
    """
    rule 5 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k,node: str
            nodes to test for orientation rule
    """
    changes = False
    for path in pag.findUncoveredCirclePaths(i, j):
        edge = pag.has_o(i, j, j) and pag.has_o(i, j, j)
        on_path = False
        if node in path and k in path:
            on_path = path.index(k) == 1 and path.index(
                node) == (len(path) - 2)
        nonadj = not pag.has_edge(i, node) and not pag.has_edge(k, j)
        if edge and on_path and nonadj:
            pag.undirect_edge(i, j)
            # print("Orienting edge {},{} with rule 5".format(i, j))
            for x in range(len(path) - 1):
                pag.undirect_edge(path[x], path[x + 1])
                # print(
                #     "Orienting edge {},{} with rule 5".format(path[x],
                #                                               path[x + 1])
                # )
                changes = True
    return changes


def rule67(pag, i, j, k):
    """
    rules 6 and 7 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k: str
            nodes to test for orientation rule
    """
    changes = False
    if pag.has_edge(i, j) and pag.has_edge(j, k):
        edge1 = pag.get_edge_data(i, j)
        edge2 = pag.get_edge_data(j, k)
        if edge1[i] == "-" and edge1[j] == "-" and edge2[j] == "o":
            pag.setTag([j, k], j, "-")
            # print("Orienting edge {},{} with rule 6".format(k, j))
            changes = True
        if (
                edge1[i] == "-"
                and edge1[j] == "o"
                and edge2[j] == "o"
                and not pag.has_edge(i, k)
        ):
            pag.setTag([j, k], j, "-")
            # print("Orienting edge {},{} with rule 7".format(k, j))
            changes = True
    return changes


def rule8(pag, i, j, k):
    """
    rule 8 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k: str
            nodes to test for orientation rule
    """
    chain1 = pag.has_fully_directed_edge(
        i, j) and pag.has_fully_directed_edge(j, k)
    chain2 = False
    edge = False
    if pag.has_edge(i, j) and pag.has_edge(i, k):
        chain2 = (
            pag.has_directed_edge(j, k)
            and pag.get_edge_data(i, j)[j] == "o"
            and pag.get_edge_data(i, j)[i] == "-"
        )
        edge = pag.get_edge_data(i, k)[i] == "o" and pag.has_directed_edge(
            i, k)
    if (chain1 or chain2) and edge:
        pag.fully_direct_edge(i, k)
        # print("Orienting edge {},{} with rule 8".format(k, i))
        return True
    return False


def rule9(pag, i, j, k, node):
    """
    rule 9 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k,node: str
            nodes to test for orientation rule
    """
    if pag.has_directed_edge(i, k) and pag.has_o(i, k, i):
        for path in nx.all_simple_paths(pag, i, k):
            if pag.isUncovered(path) and pag.isPD(path):
                if path[1] == j and path[2] == node and not pag.has_edge(j, k):
                    pag.fully_direct_edge(i, k)
                    # print("Orienting edge {},{} with rule 9".format(k, i))
                    return True
    return False


def rule10(pag, i, j, k, node):
    """
    rule 10 of edge orientation in the fci algorithm

    Parameters
    ----------
        pag: PAG
            PAG to orient edges of
        i,j,k,node: str
            nodes to test for orientation rule
    """
    changes = False
    if pag.has_directed_edge(i, k) and pag.has_o(i, k, i):
        if pag.has_fully_directed_edge(j, k) and pag.has_fully_directed_edge(node, k):
            for path1 in nx.all_simple_paths(pag, i, j):
                for path2 in nx.all_simple_paths(pag, i, node):
                    if (
                            pag.isUncovered(path1)
                            and pag.isPD(path1)
                            and pag.isUncovered(path2)
                            and pag.isPD(path2)
                    ):
                        if path1[1] != path2[1] and not pag.has_edge(
                                path1[1], path2[1]
                        ):
                            pag.fully_direct_edge(i, k)
                            # print(
                            #     "Orienting edge {},{} with rule 10".format(
                            #         k, i))
                            changes = True
    return changes
