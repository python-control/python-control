#!/usr/bin/env python
#
# freqresp_test.py - test frequency response functions
# RMM, 30 May 2016 (based on timeresp_test.py)
#
# This is a rudimentary set of tests for frequency response functions, 
# including bode plots. 

import unittest
import numpy as np
from control.statesp import StateSpace
from control.matlab import ss, tf, bode
import matplotlib.pyplot as plt

class TestFreqresp(unittest.TestCase):
   def setUp(self):
      self.A = np.matrix('1,1;0,1')
      self.C = np.matrix('1,0')
      self.omega = np.linspace(10e-2,10e2,1000)

   def test_siso(self):
      B = np.matrix('0;1')
      D = 0
      sys = StateSpace(self.A,B,self.C,D)

      # test frequency response
      frq=sys.freqresp(self.omega)

      # test bode plot
      bode(sys)

      # Convert to transfer function and test bode
      systf = tf(sys)
      bode(systf)

   def test_doubleint(self):
      # 30 May 2016, RMM: added to replicate typecast bug in freqresp.py
      A = np.matrix('0, 1; 0, 0');
      B = np.matrix('0; 1');
      C = np.matrix('1, 0');
      D = 0;
      sys = ss(A, B, C, D);
      bode(sys);

   def test_mimo(self):
      # MIMO
      B = np.matrix('1,0;0,1')
      D = np.matrix('0,0')
      sysMIMO = ss(self.A,B,self.C,D)

      frqMIMO = sysMIMO.freqresp(self.omega)
      tfMIMO = tf(sysMIMO)

      #bode(sysMIMO) # - should throw not implemented exception
      #bode(tfMIMO) # - should throw not implemented exception
      
      #plt.figure(3)
      #plt.semilogx(self.omega,20*np.log10(np.squeeze(frq[0])))

      #plt.figure(4)
      #bode(sysMIMO,self.omega)

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)

if __name__ == '__main__':
   unittest.main()
