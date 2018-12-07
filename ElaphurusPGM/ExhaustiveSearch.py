from itertools import permutations, chain, combinations
from ElaphurusPGM.BicScore import Bic
from ElaphurusPGM.Model import BayesianModel
import numpy as np
import networkx as nx


class ExhaustiveSearch(object):
    """docstring for ExhaustiveSearch"""
    def __init__(self, data, scoreMethod=None, estimator=None):
        super(ExhaustiveSearch, self).__init__()
        self.data = data
        self.nodes = list(data.columns)
        if not scoreMethod:
            self.scoreMethod = Bic(data)
        else:
            self.scoreMethod = scoreMethod

    def all_dags(self):
        edges = list(permutations(self.nodes,2))
        all_graphs = chain.from_iterable(combinations(edges,r) for r in range(len(edges)+1))
        count = 0
        for graph_edges in all_graphs:
            graph = nx.DiGraph(graph_edges)
            if nx.is_directed_acyclic_graph(graph):
                yield graph
                count += 1
        print("Totally", count, "graphs")

    def estimate(self):
        best_dag = max(self.all_dags(), key=self.scoreMethod.score)
        best_model = BayesianModel()
        best_model.add_nodes_from(sorted(best_dag.nodes))
        best_model.add_edges_from(sorted(best_dag.edges))
        return best_model
