#!/usr/bin/env python

### MUST BE CONVERTED TO A UNIT TEST!!!


# Script to test frequency response and frequency response plots like bode, nyquist and gang of 4.
# Especially need to ensure that nothing SISO is broken and that MIMO at least handles exceptions and has some default to SISO in place.


import unittest
from statesp import StateSpace
from matlab import ss, tf, bode
import numpy as np
import matplotlib.pyplot as plt

# SISO
plt.close('all')

A = np.matrix('1,1;0,1')
B = np.matrix('0;1')
C = np.matrix('1,0')
D = 0
sys = StateSpace(A,B,C,D)
#or try
#sys = ss(A,B,C,D)

# test frequency response
omega = np.linspace(10e-2,10e2,1000)
frq=sys.freqresp(omega)

# MIMO
B = np.matrix('1,0;0,1')
D = np.matrix('0,0')
sysMIMO = ss(A,B,C,D)
frqMIMO = sysMIMO.freqresp(omega)

plt.figure(1)
bode(sys)

systf = tf(sys)
tfMIMO = tf(sysMIMO)

print systf.pole()
#print tfMIMO.pole() # - should throw not implemented exception
#print tfMIMO.zero() # - should throw not implemented exception

plt.figure(2)
bode(systf)

#bode(sysMIMO) # - should throw not implemented exception
#bode(tfMIMO) # - should throw not implemented exception

#plt.figure(3)
#plt.semilogx(omega,20*np.log10(np.squeeze(frq[0])))

#plt.figure(4)
#bode(sysMIMO,omega)


def suite():
   pass
   #Uncomment this once it is a real unittest
   #return unittest.TestLoader().loadTestsFromTestCase(TestFreqRsp)
