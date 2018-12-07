import numpy as np
import copy

class DiscreteFactor(object):
    """
    DiscreteFactor
    """
    def __init__(self, variables, cardinality, values):
        super(DiscreteFactor, self).__init__()
        self.variables = list(variables)
        self.values = np.array(values, dtype=np.float)
        self.cardinality = np.array(cardinality,dtype=np.int)
        self.values = self.values.reshape(self.cardinality)

    def deepcopy(self):
        return copy.deepcopy(self)

    def scope(self):
        '''
        return the variables
        '''
        return self.variables

    def get_cardinality(self,variables=None):
        '''
        return the cardinalities of the variables
        '''
        if not variables:
            return {var: self.cardinality[self.variables.index(var)] for var in self.variables}
        else:
            return {var: self.cardinality[self.variables.index(var)] for var in variables}

    def marginalize(self, variables, inplace=True):
        '''
        marginalize the variables of the values
        '''
        phi = self if inplace else self.deepcopy()
        var_indexes = [phi.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(self.variables)))-set(var_indexes))  # sort to keep the original variabes order
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]
        
        phi.values = np.sum(phi.values, axis=tuple(var_indexes))  # sum out the marginalized variables

        if not inplace:
            return phi

    def normalize(self, inplace=True):
        '''
        normalize the whole factor
        '''
        phi = self if inplace else self.deepcopy()
        oldshape = phi.values.shape
        phi.values = phi.values.reshape(-1,phi.values.shape[-1]) / phi.values.sum(-1).flatten().reshape(-1,1)
        phi.values = phi.values.reshape(oldshape)
        if not inplace:
            return phi

    def restrict(self, values, inplace=True):
        '''
        restrict means giving an assignment to the evidences variables
            values: list of ('var':'assignment')
        '''
        phi = self if inplace else self.deepcopy()
        var_index_to_del = []
        slice_ = [slice(None)]*len(self.variables)
        for var, state in values:
            var_index = phi.variables.index(var)
            slice_[var_index] = state
            var_index_to_del.append(var_index)
        var_index_to_keep = sorted(set(range(len(phi.variables))) - set(var_index_to_del))  # sort to keep the original variabes order
        phi.variables = [phi.variables[index] for index in var_index_to_keep]
        phi.cardinality = phi.cardinality[var_index_to_keep]
        phi.values = phi.values[tuple(slice_)]

        if not inplace:
            return phi


    def product(self, phiR, inplace=True):
        '''
        return the factor product with phiR
        '''
        phi = self if inplace else self.deepcopy()
        if isinstance(phiR, (int, float)):
            phi.values *= phiR
        else:
            phiR = phiR.deepcopy()  # phiR may be modified
            # print(phi.variables, phi.values.shape)
            # print(phiR.variables, phiR.values.shape)
            # note that the result of the multiplication has a different size and ndim
            # implement multiplication by elementwise multiplication and broadcasting

            # modifying phi to add new variables
            extra_vars = set(phiR.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)]*len(phi.variables)
                slice_.extend([np.newaxis]*len(extra_vars))
                phi.values = phi.values[slice_]
                phi.variables.extend(extra_vars)

                extra_var_card = phiR.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality,[extra_var_card[var] for var in extra_vars])

            # print(phi.variables, phi.values.shape)
            # print(phiR.variables, phiR.values.shape)

            # modifying phiR to add new variables
            extra_vars = set(phi.variables) - set(phiR.variables)
            if extra_vars:
                slice_ = [slice(None)]*len(phiR.variables)
                slice_.extend([np.newaxis]*len(extra_vars))
                phiR.values = phiR.values[slice_]
                phiR.variables.extend(extra_vars)
                # cardinality is not needed

            # print(phi.variables, phi.values.shape)
            # print(phiR.variables, phiR.values.shape)

            # rearranging the axes of phiR to match phi
            # why the following code has correct performance?
            # the factor product is the same with the natural connect in the database concept, with the specification that both table has the same values in the common variables
            # so the size is the product of the cardinalities of all the variables
            # the procedure that the newaxes are broadcast to the other table is the same as that a tuple is multiplied many times and many new tuples are produced
            for axis in range(phi.values.ndim):
                exchange_index = phiR.variables.index(phi.variables[axis])
                phiR.variables[axis], phiR.variables[exchange_index] = phiR.variables[exchange_index], phiR.variables[axis]
                phiR.values = phiR.values.swapaxes(axis, exchange_index)
            # print(phi.variables, phi.values.shape)
            # print(phiR.variables, phiR.values.shape)
            phi.values = phi.values * phiR.values

        if not inplace:
            return phi

    def __mul__(self, phiR):
        return self.product(phiR, inplace=False)