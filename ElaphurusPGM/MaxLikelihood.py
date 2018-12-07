from ElaphurusPGM.CPD import CPD
from numba import jit
import numpy as np


class MaxLikelihood(object):
    """docstring for MLE"""
    def __init__(self, model, data):
        super(MaxLikelihood, self).__init__()
        self.model = model
        self.data = data
    
    def estimate_cpd(self, node):
        parents = list(self.model.predecessors(node))
        cardinalities = []
        states_name = {}
        for n in parents+[node]:
            states_name[n] = list(set(self.data[n].values))
            cardinalities.append(len(states_name[n]))
        values = np.ones(cardinalities)
        # values = np.zeros(cardinalities)
        for tup in range(len(self.data)):
            slice_ = []
            for i, n in enumerate(parents+[node]):
                slice_.append([states_name[n].index(self.data[n][tup])])
            values[slice_] += 1
        # print(values)
        # values = values / values.sum(axis=-1)
        cpd = CPD(node, cardinalities[-1], values, parents, cardinalities[:-1])
        cpd.normalize()
        return cpd

    def get_parameters(self):
        parameters = [self.estimate_cpd(node) for node in self.model.nodes]
        return parameters