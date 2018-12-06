import sys
sys.path.append('..')
import numpy as np
from ElaphurusPGM.DiscreteFactor import DiscreteFactor
phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], np.arange(12))
phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], np.arange(8))
phi1.product(phi2, inplace=True)
print(phi1.values)