"""slycot_convert_test.py - test SLICOT-based conversions

RMM, 30 Mar 2011 (based on TestSlycot from v0.4a)
"""

import numpy as np
import pytest

from control import bode, rss, ss, tf
from control.tests.conftest import slycotonly

numTests = 5
maxStates = 10
maxI = 1
maxO = 1


@pytest.fixture(scope="module")
def fixedseed():
    """Get consistent test results"""
    np.random.seed(0)


@slycotonly
@pytest.mark.usefixtures("fixedseed")
class TestSlycot:
    """Test Slycot system conversion

    TestSlycot compares transfer function and state space conversions for
    various numbers of inputs,outputs and states.
    1. Usually passes for SISO systems of any state dim, occasonally,
       there will be a dimension mismatch if the original randomly
       generated ss system is not minimal because td04ad returns a
       minimal system.

    2. For small systems with many inputs, n<<m, the tests fail
       because td04ad returns a minimal ss system which has fewer
       states than the original system. It is typical for systems
       with many more inputs than states to have extraneous states.

    3. For systems with larger dimensions, n~>5 and with 2 or more
       outputs the conversion to statespace (td04ad) intermittently
       results in an equivalent realization of higher order than the
       original tf order. We think this has to do with minimu
       realization tolerances in the Fortran. The algorithm doesn't
       recognize that two denominators are identical and so it
       creates a system with nearly duplicate eigenvalues and
       double the state dimension. This should not be a problem in
       the python-control usage because the common_den() method finds
       repeated roots within a tolerance that we specify.

    Matlab: Matlab seems to force its statespace system output to
    have order less than or equal to the order of denominators provided,
    avoiding the problem of very large state dimension we describe in 3.
    It does however, still have similar problems with pole/zero
    cancellation such as we encounter in 2, where a statespace system
    may have fewer states than the original order of transfer function.
    """

    @pytest.fixture
    def verbose(self):
        """Set to True and switch off pytest stdout capture to print info"""
        return False

    @pytest.mark.parametrize("testNum", np.arange(numTests) + 1)
    @pytest.mark.parametrize("inputs", np.arange(maxI) + 1)
    @pytest.mark.parametrize("outputs", np.arange(maxO) + 1)
    @pytest.mark.parametrize("states", np.arange(maxStates) + 1)
    def testTF(self, states, outputs, inputs, testNum, verbose):
        """Test transfer function conversion.

        Directly tests the functions tb04ad and td04ad through direct
        comparison of transfer function coefficients.
        Similar to convert_test, but tests at a lower level.
        """
        from slycot import tb04ad, td04ad

        ssOriginal = rss(states, outputs, inputs)
        if (verbose):
            print('====== Original SS ==========')
            print(ssOriginal)
            print('states=', states)
            print('inputs=', inputs)
            print('outputs=', outputs)

        tfOriginal_Actrb, tfOriginal_Bctrb, tfOriginal_Cctrb,\
            tfOrigingal_nctrb, tfOriginal_index,\
            tfOriginal_dcoeff, tfOriginal_ucoeff =\
            tb04ad(states, inputs, outputs,
                   ssOriginal.A, ssOriginal.B,
                   ssOriginal.C, ssOriginal.D, tol1=0.0)

        ssTransformed_nr, ssTransformed_A, ssTransformed_B,\
            ssTransformed_C, ssTransformed_D\
            = td04ad('R', inputs, outputs, tfOriginal_index,
                     tfOriginal_dcoeff, tfOriginal_ucoeff,
                     tol=0.0)

        tfTransformed_Actrb, tfTransformed_Bctrb,\
            tfTransformed_Cctrb, tfTransformed_nctrb,\
            tfTransformed_index, tfTransformed_dcoeff,\
            tfTransformed_ucoeff = tb04ad(
                ssTransformed_nr, inputs, outputs,
                ssTransformed_A, ssTransformed_B,
                ssTransformed_C, ssTransformed_D, tol1=0.0)
        # print('size(Trans_A)=',ssTransformed_A.shape)
        if (verbose):
            print('===== Transformed SS ==========')
            print(ss(ssTransformed_A, ssTransformed_B,
                     ssTransformed_C, ssTransformed_D))
            # print('Trans_nr=',ssTransformed_nr
            # print('tfOrig_index=',tfOriginal_index)
            # print('tfOrig_ucoeff=',tfOriginal_ucoeff)
            # print('tfOrig_dcoeff=',tfOriginal_dcoeff)
            # print('tfTrans_index=',tfTransformed_index)
            # print('tfTrans_ucoeff=',tfTransformed_ucoeff)
            # print('tfTrans_dcoeff=',tfTransformed_dcoeff)
        # Compare the TF directly, must match
        # numerators
        # TODO test failing!
        # np.testing.assert_array_almost_equal(
        #    tfOriginal_ucoeff, tfTransformed_ucoeff, decimal=3)
        # denominators
        # np.testing.assert_array_almost_equal(
        #    tfOriginal_dcoeff, tfTransformed_dcoeff, decimal=3)

    @pytest.mark.usefixtures("legacy_plot_signature")
    @pytest.mark.parametrize("testNum", np.arange(numTests) + 1)
    @pytest.mark.parametrize("inputs", np.arange(1) + 1) # SISO only
    @pytest.mark.parametrize("outputs", np.arange(1) + 1) # SISO only
    @pytest.mark.parametrize("states", np.arange(maxStates) + 1)
    def testFreqResp(self, states, outputs, inputs, testNum, verbose):
        """Compare bode responses.

        Compare the bode reponses of the SS systems and TF systems to the
        original SS. They generally are different realizations but have same
        freq resp. Currently this test may only be applied to SISO systems.
        """
        from slycot import tb04ad, td04ad

        ssOriginal = rss(states, outputs, inputs)

        tfOriginal_Actrb, tfOriginal_Bctrb, tfOriginal_Cctrb,\
            tfOrigingal_nctrb, tfOriginal_index,\
            tfOriginal_dcoeff, tfOriginal_ucoeff = tb04ad(
                states, inputs, outputs, ssOriginal.A,
                ssOriginal.B, ssOriginal.C, ssOriginal.D,
                tol1=0.0)

        ssTransformed_nr, ssTransformed_A, ssTransformed_B,\
            ssTransformed_C, ssTransformed_D\
            = td04ad('R', inputs, outputs, tfOriginal_index,
                     tfOriginal_dcoeff, tfOriginal_ucoeff,
                     tol=0.0)

        tfTransformed_Actrb, tfTransformed_Bctrb,\
            tfTransformed_Cctrb, tfTransformed_nctrb,\
            tfTransformed_index, tfTransformed_dcoeff,\
            tfTransformed_ucoeff = tb04ad(
                ssTransformed_nr, inputs, outputs,
                ssTransformed_A, ssTransformed_B,
                ssTransformed_C, ssTransformed_D,
                tol1=0.0)

        numTransformed = np.array(tfTransformed_ucoeff)
        denTransformed = np.array(tfTransformed_dcoeff)
        numOriginal = np.array(tfOriginal_ucoeff)
        denOriginal = np.array(tfOriginal_dcoeff)

        ssTransformed = ss(ssTransformed_A,
                           ssTransformed_B,
                           ssTransformed_C,
                           ssTransformed_D)
        for inputNum in range(inputs):
            for outputNum in range(outputs):
                [ssOriginalMag, ssOriginalPhase, freq] =\
                    bode(ssOriginal, plot=False)
                [tfOriginalMag, tfOriginalPhase, freq] =\
                    bode(tf(numOriginal[outputNum][inputNum],
                            denOriginal[outputNum]),
                         plot=False)
                [ssTransformedMag, ssTransformedPhase, freq] =\
                    bode(ssTransformed,
                         freq,
                         plot=False)
                [tfTransformedMag, tfTransformedPhase, freq] =\
                    bode(tf(numTransformed[outputNum][inputNum],
                            denTransformed[outputNum]),
                         freq,
                         plot=False)
                # print('numOrig=',
                #  numOriginal[outputNum][inputNum])
                # print('denOrig=',
                #  denOriginal[outputNum])
                # print('numTrans=',
                #  numTransformed[outputNum][inputNum])
                # print('denTrans=',
                #  denTransformed[outputNum])
                np.testing.assert_array_almost_equal(
                    ssOriginalMag, tfOriginalMag, decimal=3)
                np.testing.assert_array_almost_equal(
                    ssOriginalPhase, tfOriginalPhase,
                    decimal=3)
                np.testing.assert_array_almost_equal(
                    ssOriginalMag, ssTransformedMag, decimal=3)
                np.testing.assert_array_almost_equal(
                    ssOriginalPhase, ssTransformedPhase,
                    decimal=3)
                np.testing.assert_array_almost_equal(
                    tfOriginalMag, tfTransformedMag, decimal=3)
                np.testing.assert_array_almost_equal(
                    tfOriginalPhase, tfTransformedPhase,
                    decimal=2)

