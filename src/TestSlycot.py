#!/usr/bin/env python


import numpy as np
from slycot import tb04ad, td04ad
import matlab
import unittest


class TestSlycot(unittest.TestCase):
    def setUp(self):
        """Define some test parameters."""
        self.numTests = 1
        self.maxStates = 10 #seems to fail rarely with 4, sometimes with 5, frequently with 6. Could it be a problem with the subsystems?
        self.maxIO = 10 
     
    def testTF(self):
        """ Directly tests the functions tb04ad and td04ad through direct comparison of transfer function coefficients.
            Similar to TestConvert, but tests at a lower level.
        """
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO+1):
                for outputs in range(1, self.maxIO+1):
                    for testNum in range(self.numTests):
                        
                        ssOriginal = matlab.rss(states, inputs, outputs)
                        
                        print '====== Original SS =========='
                        print ssOriginal
                        print 'states=',states
                        print 'inputs=',inputs
                        print 'outputs=',outputs
                        
                        
                        tfOriginal_Actrb, tfOriginal_Bctrb, tfOriginal_Cctrb, tfOrigingal_nctrb, tfOriginal_index,\
                            tfOriginal_dcoeff, tfOriginal_ucoeff = tb04ad('R',states,inputs,outputs,\
                            ssOriginal.A,ssOriginal.B,ssOriginal.C,ssOriginal.D,tol1=1e-10)
                        
                        ssTransformed_nr, ssTransformed_A, ssTransformed_B, ssTransformed_C, ssTransformed_D\
                            = td04ad('R',inputs,outputs,tfOriginal_index,tfOriginal_dcoeff,tfOriginal_ucoeff,tol=1e-8)
                        
                        tfTransformed_Actrb, tfTransformed_Bctrb, tfTransformed_Cctrb, tfTransformed_nctrb,\
                            tfTransformed_index, tfTransformed_dcoeff, tfTransformed_ucoeff = tb04ad('R',\
                            ssTransformed_nr,inputs,outputs,ssTransformed_A, ssTransformed_B, ssTransformed_C,\
                            ssTransformed_D,tol1=1e-10)
                        print 'size(Trans_A)=',ssTransformed_A.shape
                        print 'Trans_nr=',ssTransformed_nr                      
                        print 'tfOrig_index=',tfOriginal_index
                        print 'tfOrig_ucoeff=',tfOriginal_ucoeff
                        print 'tfOrig_dcoeff=',tfOriginal_dcoeff
                        print 'tfTrans_index=',tfTransformed_index
                        print 'tfTrans_ucoeff=',tfTransformed_ucoeff
                        print 'tfTrans_dcoeff=',tfTransformed_dcoeff
                       #Compare the TF directly, must match
                        #numerators
                        np.testing.assert_array_almost_equal(tfOriginal_ucoeff,tfTransformed_ucoeff,decimal=3)
                        #denominators
                        np.testing.assert_array_almost_equal(tfOriginal_dcoeff,tfTransformed_dcoeff,decimal=3)
                           
    def testFreqResp(self):
        """Compare the bode reponses of the SS systems and TF systems to the original SS
           They generally are different realizations but have same freq resp. 
           Currently this test may only be applied to SISO systems.
                      
        for states in range(1,self.maxStates):
            for testNum in range(self.numTests):                       
                for inputs in range(1,self.maxIO+1):
                    for outputs in range(1,self.maxIO+1):       
                        ssOriginal = matlab.rss(states, inputs, outputs)
                        
                        tfOriginal_Actrb, tfOriginal_Bctrb, tfOriginal_Cctrb, tfOrigingal_nctrb, tfOriginal_index,\
                            tfOriginal_dcoeff, tfOriginal_ucoeff = tb04ad('R',states,inputs,outputs,\
                            ssOriginal.A,ssOriginal.B,ssOriginal.C,ssOriginal.D,tol1=1e-10)
                        
                        ssTransformed_nr, ssTransformed_A, ssTransformed_B, ssTransformed_C, ssTransformed_D\
                            = td04ad('R',inputs,outputs,tfOriginal_index,tfOriginal_dcoeff,tfOriginal_ucoeff,tol=1e-8)
                        
                        tfTransformed_Actrb, tfTransformed_Bctrb, tfTransformed_Cctrb, tfTransformed_nctrb,\
                            tfTransformed_index, tfTransformed_dcoeff, tfTransformed_ucoeff = tb04ad('R',\
                            ssTransformed_nr,inputs,outputs,ssTransformed_A, ssTransformed_B, ssTransformed_C,\
                            ssTransformed_D,tol1=1e-10)

                        numTransformed = np.array(tfTransformed_ucoeff)
                        denTransformed = np.array(tfTransformed_dcoeff)
                        numOriginal = np.array(tfOriginal_ucoeff)
                        denOriginal = np.array(tfOriginal_dcoeff)
                                              
                        ssTransformed = matlab.ss(ssTransformed_A,ssTransformed_B,ssTransformed_C,ssTransformed_D)
                        for inputNum in range(inputs):
                            for outputNum in range(outputs):
                                #[ssOriginalMag,ssOriginalPhase,freq] = matlab.bode(ssOriginal,Plot=False) 
                                [tfOriginalMag,tfOriginalPhase,freq] = matlab.bode(matlab.tf(numOriginal[outputNum][inputNum],denOriginal[outputNum]),Plot=False)
                                #[ssTransformedMag,ssTransformedPhase,freq] = matlab.bode(ssTransformed,freq,Plot=False)
                                [tfTransformedMag,tfTransformedPhase,freq] = matlab.bode(matlab.tf(numTransformed[outputNum][inputNum],denTransformed[outputNum]),freq,Plot=False)
                                print 'numOrig=',numOriginal[outputNum][inputNum]
                                print 'denOrig=',denOriginal[outputNum]
                                print 'numTrans=',numTransformed[outputNum][inputNum]
                                print 'denTrans=',denTransformed[outputNum]
                                #np.testing.assert_array_almost_equal(ssOriginalMag,tfOriginalMag,decimal=3)
                                #np.testing.assert_array_almost_equal(ssOriginalPhase,tfOriginalPhase,decimal=3)       
                                #np.testing.assert_array_almost_equal(ssOriginalMag,ssTransformedMag,decimal=3)
                                #np.testing.assert_array_almost_equal(ssOriginalPhase,ssTransformedPhase,decimal=3)
                                #np.testing.assert_array_almost_equal(tfOriginalMag,tfTransformedMag,decimal=3)
                                np.testing.assert_array_almost_equal(tfOriginalPhase,tfTransformedPhase,decimal=2)
        """                
#These are here for once the above is made into a unittest.
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSlycot)

if __name__=='__main__':
    unittest.main()

