from ElaphurusPGM.DiscreteFactor import DiscreteFactor
import numpy as np

class CPD(DiscreteFactor):
    """
    Conditional probability distribution
    """
    def __init__(self, variable, variable_card, values, parent=None, parent_card=None):
        self.variable = variable
        self.variable_card = variable_card
        variables = [variable] + (list(parent) if parent else [])
        cardinality = [variable_card] + (list(parent_card) if parent_card else [])
        values = np.array(values)
        super(CPD, self).__init__(variables,cardinality,values.flatten(order='C'))

    def get_parents(self):
        return self.variables[:0:-1]

    def to_factor(self):
        return DiscreteFactor(self.variables, self.cardinality, self.values)