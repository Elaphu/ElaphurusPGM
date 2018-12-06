from collections import defaultdict
from ElaphurusPGM.FactorBase import factor_product
import copy
import numpy as np


class Inference(object):
    """
    Inference
    """
    def __init__(self, model):
        self.model = model
        self.variables = model.nodes()
        self.cardinality = {}
        self.factors = []
        for node in model.nodes():
            cpd = model.get_cpds(node)
            self.cardinality[node] = cpd.variable_card
            cpd = cpd.to_factor()
            self.factors.append(cpd)

class VariableElimination(Inference):
    """
    value elimination
    """
    def __init__(self, model):
        super(VariableElimination, self).__init__(model)
    
    def get_elimination_order(self, variables, evidence):
        '''
        TODO: find the order using heuristics
        Now: polytree and minfill
        '''
        # print("evidence : ", evidence)
        # print("self.variables : ", self.variables)
        # print("variables : ", variables)
        order = list(set(self.variables)-set(variables)-(set(evidence.keys()) if evidence else set([])))
        # np.random.shuffle(order)
        order.sort(key=lambda node: self.model.degree(node))
        return list(order)

    def query(self, variables, evidence=None, elimination_order=None):
        if not variables:
            return self.factors

        factors = copy.deepcopy(self.factors)

        working_factors = defaultdict(list)

        if not elimination_order:
            ret_order = True
            elimination_order = self.get_elimination_order(variables, evidence)

        # print(elimination_order+variables)

        # no query variables are in the elimination_order

        # restrict all the evidence variables
        if evidence:
            for evidence_var in evidence:
                for i, factor in enumerate(factors):
                    if evidence_var in factor.variables:
                        factor_restricted = factor.restrict([(evidence_var, evidence[evidence_var])], inplace=False)
                        factors[i] = factor_restricted

        # for f in factors:
            # print(f.variables,f.values)

        for factor in factors:
            for var in elimination_order+variables:
                if var in factor.variables:
                    working_factors[var].append(factor)
                    break

        # for key in elimination_order + variables:
            # print(key)
            # for v in working_factors[key]:
                # print(v.variables,v.values)

        for var in elimination_order:
            # Remove all the factors according to the elimination_order
            phi = factor_product(*working_factors[var])
            phi.marginalize([var], inplace=True)
            del working_factors[var]
            for var2 in elimination_order+variables:
                if var2 in phi.variables:
                    working_factors[var2].append(phi)
                    break
            # for key in elimination_order+variables:
                # print(key)
                # for v in working_factors[key]:
                    # print(v.variables,v.values)

        final_distribution = []

        # print("final")

        for node in working_factors:
            if working_factors[node]:
                final_distribution.extend(working_factors[node])
                # print(node)
                # for val in working_factors[node]:
                    # print(val.values)

        query_var_factor = {}
        final_phi = factor_product(*final_distribution)

        # print("final phi", final_phi.values)

        for query_var in variables:
            query_var_factor[query_var] = final_phi.marginalize(list(set(variables)-set([query_var])),inplace=False).normalize(inplace=False)

        if len(variables) > 1:
            query_var_factor['JDP'] = final_phi

        if ret_order:
            return query_var_factor, elimination_order
        else:
            return query_var_factor


