from control.matlab import *
import numpy as np
from control.frdata import FRD as frd

sys = ss(np.matrix('-2 0; 0 -1'), np.eye(2), np.eye(2), np.zeros((2,2)))
print sys

# convert to frd, 1 frequency
fr = frd(sys, [1], smooth=False)
print fr

# the matching matrix
frm = np.matrix([[0.4-0.2j, 0], [0, 0.5-0.5j]])
print frm

# feedback of the system itself
sys2 = sys.feedback(np.matrix('0 1; 3 0'))
print sys2

# and the matching fr
fr2 = frd(sys2, [1], smooth=False)
print fr2

fr2b = fr.feedback([[0, 1], [3, 0]])
print fr2b

# frequency response from the matrix should be
frm*(np.eye(2)+np.matrix('0 1.0; 3 0')*frm).I
print frm

# one with 3 out and 2 inputs, 3 states
bsys = ss(np.matrix('-2.0 0 0; 0 -1 1; 0 0 -3'), np.matrix('1.0 0; 0 0; 0 1'), np.eye(3), np.zeros((3,2)))
print bsys

# convert to frd, 1 frequency
bfr = frd(bsys, [1])
print bfr

# the matching matrix
bfrm = np.matrix('0.4-0.2j 0;0 0.1-0.2j; 0  0.3-0.1j')
print bfrm

K = np.matrix('1 0.3 0; 0.1 0 0')
bsys2 = bsys.feedback(K)
print bsys2

# and the matching fr
bfr2 = frd(bsys2, [1])
print bfr2

bfr2b = bfr.feedback( K)
print bfr2b

# frequency response from the matrix should be
print bfrm*(np.eye(2)+K*bfrm).I
