from control import TransferFunction
from sys import float_info
import numpy as np

s = TransferFunction([1, 0], [1])

h = (s+1)*(s+2)/(s+2)/(s**2+s+1)
tol = None

sqrt_eps = np.sqrt(float_info.epsilon)

for i in range(h.inputs):
    for j in range(h.outputs):
        newzeros = []
        zeros = np.roots(h.num[i][j])
        poles = np.roots(h.den[i][j])
        # check all zeros
        for z in zeros:
            t = tol or \
                1000 * max(float_info.epsilon, abs(z) * sqrt_eps)
            idx = np.where(abs(z - poles) < t)[0]
            print idx
            if len(idx):
                print "found match %s" % (abs(z - poles) < t)
                poles = np.delete(poles, idx[0])
            else:
                newzeros.append(z)
        print newzeros
        print poles
            
                
        

        
