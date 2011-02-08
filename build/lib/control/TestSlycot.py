#!/usr/bin/env python

#### THIS MUST BE MADE INTO A UNITTEST TO BE PART OF THE TESTING FUNCTIONS!!!!


import numpy as np
from slycot import tb04ad, td04ad
import matlab

numTests = 1
maxStates = 3
maxIO = 3

for states in range(1, maxStates):
            for inputs in range(1, maxIO):
                for outputs in range(1, maxIO):
                    sys1 = matlab.rss(states, inputs, outputs)
                    print "sys1"
                    print sys1

                    sys2 = tb04ad(states,inputs,outputs,sys1.A,sys1.B,sys1.C,sys1.D,outputs,outputs,inputs)
                    print "sys2"
                    print sys2

                    ldwork = 100*max(1,states+max(states,max(3*inputs,3*outputs)))
                    dwork = np.zeros(ldwork)
                    sys3 = td04ad(inputs,outputs,sys2[4],sys2[5],sys2[6])
                    #sys3 = td04ad(inputs,outputs,sys2[4],sys2[5],sys2[6],ldwork)
                    print "sys3"
                    print sys3

                    sys4 = tb04ad(states,inputs,outputs,sys3[1][0:states,0:states],sys3[2][0:states,0:inputs],sys3[3][0:outputs,0:states],sys3[4],outputs,outputs,inputs)
                    print "sys4"
                    print sys4

#These are here for once the above is made into a unittest.
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestSlycot)

if __name__=='__main__':
	unittest.main()

