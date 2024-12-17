import pytest
import networkx as nx

from causalexplain.common.utils import correct_edge_from_prior

class TestCorrectEdgeFromPrior:

    """
        a ---> b
       ^ \     |
      /   v    v
      c-->d    e
          |   ^
          v  /
          f
    """
    dag = nx.DiGraph()
    dag.add_edge('a', 'b')
    dag.add_edge('c', 'a')
    dag.add_edge('a', 'd')
    dag.add_edge('c', 'd')
    dag.add_edge('b', 'e')
    dag.add_edge('d', 'f')
    dag.add_edge('f', 'e')
    prior = [['a', 'b'], ['c', 'd', 'e'], ['f']]


    def test_both_in_top_list(self):
        assert correct_edge_from_prior(self.dag, 'a', 'b', self.prior, False) ==  1
        assert not self.dag.has_edge('a', 'b')

    def test_v_is_before_u(self):
        # These edges violate the prior order.
        assert correct_edge_from_prior(self.dag, 'c', 'a', self.prior, True) == -1
        assert correct_edge_from_prior(self.dag, 'f', 'e', self.prior, True) == -1
        # Check that 'dag' is changed
        assert not self.dag.has_edge('a', 'c')
        assert not self.dag.has_edge('f', 'e')

    def test_u_is_before_v(self):
        # These edges are in the correct order.
        assert correct_edge_from_prior(self.dag, 'a', 'd', self.prior, False) == 1
        assert correct_edge_from_prior(self.dag, 'd', 'f', self.prior, False) == 1
        assert correct_edge_from_prior(self.dag, 'b', 'e', self.prior, False) == 1

    def test_not_in_clear_order(self):
        assert correct_edge_from_prior(self.dag, 'c', 'd', self.prior, False) == 0

    def test_edge_not_present(self):
        assert correct_edge_from_prior(self.dag, 'a', 'h', self.prior, False) == 0
