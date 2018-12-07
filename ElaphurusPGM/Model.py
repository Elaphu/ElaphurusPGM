import networkx as nx 

class BayesianModel(nx.DiGraph):
    """docstring for BayesianModel"""
    def __init__(self, edge=None):
        super(BayesianModel, self).__init__(edge)

    def add_node(self, node, weight):
        super(BayesianModel, self).add_node(node, weight=weight)

    def add_edge(self, e_from, e_end, weight):
        super(BayesianModel, self).add_edge(e_from, e_end, weight=weight)

    def add_cpds(self, cpds):
        self.cpds = cpds

    def get_cpds(self, node=None):
        if node:
            for cpd in self.cpds:
                if cpd.variable == node:
                    return cpd
        else:
            return self.cpds