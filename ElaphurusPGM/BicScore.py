from numba import jit
import math


class Bic(object):
    def __init__(self, data):
        self.data =data

    @jit
    def score(self, dag):
        scores = math.log2(len(self.data))*(len(dag.nodes())+len(dag.edges)*2)
        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            counts = count_data_size(self.data, [node]+parents)
            for count in counts:
                # print(count)
                scores -= math.log2(count*len([node]+parents))
        return scores

@jit
def count_data_size(data, vars):
    '''
    return a list contains the histgram of the vars in data
    '''
    newData = data.iloc[:][vars]
    newDd = newData.drop_duplicates()
    counts = []
    for tup in range(len(newDd)):
        counts.append(0)
        for tup2 in range(len(newData)):
            # print(newData.iloc[tup2][:])
            # print(newDd.iloc[tup][:])
            if (newData.iloc[tup2][:] == newDd.iloc[tup][:]).all():
                counts[-1] += 1
    return counts