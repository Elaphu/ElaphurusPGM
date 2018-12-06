from functools import reduce

def factor_product(*args):
    '''
    factors multiplied one by one
    '''
    return reduce(lambda phi1,phi2: phi1*phi2, args)