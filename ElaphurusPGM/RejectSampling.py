from ElaphurusPGM.VE import Inference
from numba import jit
import networkx as nx
import numpy as np
import pandas as pd


class BayesianModelSampling(Inference):
    """
    BayesianModelSampling
    """
    def __init__(self, model):
        super(BayesianModelSampling, self).__init__(model)
        self.topological_order = list(nx.topological_sort(model))
        # print("topology",self.topological_order)
    
    def prior_sample(self, size=1, datatype="DataFrame"):
        '''
        Generate prior samples
        '''
        types = [(var_name, 'int') for var_name in self.topological_order]
        sampled = np.zeros(size, dtype=types).view(np.recarray)

        for i in range(size):
            # print(i)
            for node in self.topological_order:
                cpd = self.model.get_cpds(node)
                states = range(self.cardinality[node])
                parent = cpd.variables[:0:-1]
                if parent:
                    slice_ = [slice(None)]*len(cpd.variables)
                    for p in parent:
                        slice_[cpd.variables.index(p)] = sampled[i][p]
                    weights = cpd.values[slice_]
                else:
                    weights = cpd.values
                # print(node,weights)
                sampled[i][node] = np.random.choice(states, 1, p=weights)

        if datatype == "DataFrame":
            return pd.DataFrame.from_records(sampled)
        elif datatype == "recarray":
            return sampled

    def reject_sample(self,size=1, evidence=None):
        if not evidence:
            return self.prior_sample(size)
        i = 0
        # total = 0
        types = [(var_name, 'int') for var_name in self.topological_order]
        sampled = np.zeros(0, dtype=types).view(np.recarray)
        while i < size:
            _sampled = self.prior_sample(int(size/2),"recarray")
            # total += 1
            for j in range(len(_sampled)):
                consistent = True
                for node, val in evidence.items():
                    if _sampled[j][node] != val:
                        consistent = False
                if consistent:
                    # print(sampled)
                    # print(_sampled)
                    sampled = np.append(sampled,_sampled[j])
                    i += 1
            # if total % size == 0:
            #     print(total)
            #     print(i)
        return pd.DataFrame.from_records(sampled)