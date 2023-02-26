"""iosys_test.py - test input/output system operations

RMM, 17 Apr 2019

This test suite checks to make sure that basic input/output class
operations are working.  It doesn't do exhaustive testing of
operations on input/output systems.  Separate unit tests should be
created for that purpose.
"""

import re
import warnings
import pytest

import numpy as np
from math import sqrt

import control as ct
from control import iosys as ios
from control.tests.conftest import matrixfilter


class TestIOSys:

    @pytest.fixture
    def tsys(self):
        class TSys:
            pass
        T = TSys()
        """Return some test systems"""
        # Create a single input/single output linear system
        T.siso_linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]])

        # Create a multi input/multi output linear system
        T.mimo_linsys1 = ct.StateSpace(
            [[-1, 1], [0, -2]], [[1, 0], [0, 1]],
            [[1, 0], [0, 1]], np.zeros((2, 2)))

        # Create a multi input/multi output linear system
        T.mimo_linsys2 = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0, 1], [1, 0]],
            [[1, 0], [0, 1]], np.zeros((2, 2)))

        # Create a static gain linear system
        T.staticgain = ct.StateSpace([], [], [], 1)

        # Create simulation parameters
        T.T = np.linspace(0, 10, 100)
        T.U = np.sin(T.T)
        T.X0 = [0, 0]

        return T

    def test_linear_iosys(self, tsys):
        # Create an input/output system from the linear system
        linsys = tsys.siso_linsys
        iosys = ios.LinearIOSystem(linsys).copy()

        # Make sure that the right hand side matches linear system
        for x, u in (([0, 0], 0), ([1, 0], 0), ([0, 1], 0), ([0, 0], 1)):
            np.testing.assert_array_almost_equal(
                np.reshape(iosys._rhs(0, x, u), (-1, 1)),
                linsys.A @ np.reshape(x, (-1, 1)) + linsys.B * u)

        # Make sure that simulations also line up
        T, U, X0 = tsys.T, tsys.U, tsys.X0
        lti_t, lti_y = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y, atol=0.002, rtol=0.)

        # Make sure that a static linear system has dt=None
        # and otherwise dt is as specified
        assert ios.LinearIOSystem(tsys.staticgain).dt is None
        assert ios.LinearIOSystem(tsys.staticgain, dt=.1).dt == .1

    def test_tf2io(self, tsys):
        # Create a transfer function from the state space system
        linsys = tsys.siso_linsys
        tfsys = ct.ss2tf(linsys)
        iosys = ct.tf2io(tfsys)

        # Verify correctness via simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0
        lti_t, lti_y = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y, atol=0.002, rtol=0.)

        # Make sure that non-proper transfer functions generate an error
        tfsys = ct.tf('s')
        with pytest.raises(ValueError):
            iosys=ct.tf2io(tfsys)

    def test_ss2io(self, tsys):
        # Create an input/output system from the linear system
        linsys = tsys.siso_linsys
        iosys = ct.ss2io(linsys)
        np.testing.assert_allclose(linsys.A, iosys.A)
        np.testing.assert_allclose(linsys.B, iosys.B)
        np.testing.assert_allclose(linsys.C, iosys.C)
        np.testing.assert_allclose(linsys.D, iosys.D)

        # Try adding names to things
        iosys_named = ct.ss2io(linsys, inputs='u', outputs='y',
                               states=['x1', 'x2'], name='iosys_named')
        assert iosys_named.find_input('u') == 0
        assert iosys_named.find_input('x') is None
        assert iosys_named.find_output('y') == 0
        assert iosys_named.find_output('u') is None
        assert iosys_named.find_state('x0') is None
        assert iosys_named.find_state('x1') == 0
        assert iosys_named.find_state('x2') == 1
        np.testing.assert_allclose(linsys.A, iosys_named.A)
        np.testing.assert_allclose(linsys.B, iosys_named.B)
        np.testing.assert_allclose(linsys.C, iosys_named.C)
        np.testing.assert_allclose(linsys.D, iosys_named.D)

    def test_iosys_unspecified(self, tsys):
        """System with unspecified inputs and outputs"""
        sys = ios.NonlinearIOSystem(secord_update, secord_output)
        np.testing.assert_raises(TypeError, sys.__mul__, sys)

    def test_iosys_print(self, tsys, capsys):
        """Make sure we can print various types of I/O systems"""
        # Send the output to /dev/null

        # Simple I/O system
        iosys = ct.ss2io(tsys.siso_linsys)
        print(iosys)

        # I/O system without ninputs, noutputs
        ios_unspecified = ios.NonlinearIOSystem(secord_update, secord_output)
        print(ios_unspecified)

        # I/O system with derived inputs and outputs
        ios_linearized = ios.linearize(ios_unspecified, [0, 0], [0])
        print(ios_linearized)

    @pytest.mark.parametrize("ss", [ios.NonlinearIOSystem, ct.ss])
    def test_nonlinear_iosys(self, tsys, ss):
        # Create a simple nonlinear I/O system
        nlsys = ios.NonlinearIOSystem(predprey)
        T = tsys.T

        # Start by simulating from an equilibrium point
        X0 = [0, 0]
        ios_t, ios_y = ios.input_output_response(nlsys, T, 0, X0)
        np.testing.assert_array_almost_equal(ios_y, np.zeros(np.shape(ios_y)))

        # Now simulate from a nonzero point
        X0 = [0.5, 0.5]
        ios_t, ios_y = ios.input_output_response(nlsys, T, 0, X0)

        #
        # Simulate a linear function as a nonlinear function and compare
        #
        # Create a single input/single output linear system
        linsys = tsys.siso_linsys

        # Create a nonlinear system with the same dynamics
        nlupd = lambda t, x, u, params: \
            np.reshape(linsys.A @ np.reshape(x, (-1, 1))
                       + linsys.B @ np.reshape(u, (-1, 1)),
                       (-1,))
        nlout = lambda t, x, u, params: \
            np.reshape(linsys.C @ np.reshape(x, (-1, 1))
                       + linsys.D @ np.reshape(u, (-1, 1)),
                       (-1,))
        nlsys = ios.NonlinearIOSystem(nlupd, nlout, inputs=1, outputs=1)

        # Make sure that simulations also line up
        T, U, X0 = tsys.T, tsys.U, tsys.X0
        lti_t, lti_y = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(nlsys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    @pytest.fixture
    def kincar(self):
        # Create a simple nonlinear system to check (kinematic car)
        def kincar_update(t, x, u, params):
            return np.array([np.cos(x[2]) * u[0], np.sin(x[2]) * u[0], u[1]])

        def kincar_output(t, x, u, params):
            return np.array([x[0], x[1]])

        return ios.NonlinearIOSystem(
            kincar_update, kincar_output,
            inputs = ['v', 'phi'],
            outputs = ['x', 'y'],
            states = ['x', 'y', 'theta'])

    def test_linearize(self, tsys, kincar):
        # Create a single input/single output linear system
        linsys = tsys.siso_linsys
        iosys = ios.LinearIOSystem(linsys)

        # Linearize it and make sure we get back what we started with
        linearized = iosys.linearize([0, 0], 0)
        np.testing.assert_array_almost_equal(linsys.A, linearized.A)
        np.testing.assert_array_almost_equal(linsys.B, linearized.B)
        np.testing.assert_array_almost_equal(linsys.C, linearized.C)
        np.testing.assert_array_almost_equal(linsys.D, linearized.D)

        # Create a simple nonlinear system to check (kinematic car)
        iosys = kincar
        linearized = iosys.linearize([0, 0, 0], [0, 0])
        np.testing.assert_array_almost_equal(linearized.A, np.zeros((3,3)))
        np.testing.assert_array_almost_equal(
            linearized.B, [[1, 0], [0, 0], [0, 1]])
        np.testing.assert_array_almost_equal(
            linearized.C, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_almost_equal(linearized.D, np.zeros((2,2)))

    @pytest.mark.usefixtures("editsdefaults")
    def test_linearize_named_signals(self, kincar):
        # Full form of the call
        linearized = kincar.linearize([0, 0, 0], [0, 0], copy_names=True,
                                      name='linearized')
        assert linearized.name == 'linearized'
        assert linearized.find_input('v') == 0
        assert linearized.find_input('phi') == 1
        assert linearized.find_output('x') == 0
        assert linearized.find_output('y') == 1
        assert linearized.find_state('x') == 0
        assert linearized.find_state('y') == 1
        assert linearized.find_state('theta') == 2

        # If we copy signal names w/out a system name, append '$linearized'
        linearized = kincar.linearize([0, 0, 0], [0, 0], copy_names=True)
        assert linearized.name == kincar.name + '$linearized'

        # If copy is False, signal names should not be copied
        lin_nocopy = kincar.linearize(0, 0, copy_names=False)
        assert lin_nocopy.find_input('v') is None
        assert lin_nocopy.find_output('x') is None
        assert lin_nocopy.find_state('x') is None

        # if signal names are provided, they should override those of kincar
        linearized_newnames = kincar.linearize([0, 0, 0], [0, 0],
            name='linearized',
            copy_names=True, inputs=['v2', 'phi2'], outputs=['x2','y2'])
        assert linearized_newnames.name == 'linearized'
        assert linearized_newnames.find_input('v2') == 0
        assert linearized_newnames.find_input('phi2') == 1
        assert linearized_newnames.find_input('v') is None
        assert linearized_newnames.find_input('phi') is None
        assert linearized_newnames.find_output('x2') == 0
        assert linearized_newnames.find_output('y2') == 1
        assert linearized_newnames.find_output('x') is None
        assert linearized_newnames.find_output('y') is None

        # Test legacy version as well
        ct.use_legacy_defaults('0.8.4')
        ct.config.use_numpy_matrix(False)       # np.matrix deprecated
        linearized = kincar.linearize([0, 0, 0], [0, 0], copy_names=True)
        assert linearized.name == kincar.name + '_linearized'

    def test_connect(self, tsys):
        # Define a couple of (linear) systems to interconnection
        linsys1 = tsys.siso_linsys
        iosys1 = ios.LinearIOSystem(linsys1, name='iosys1')
        linsys2 = tsys.siso_linsys
        iosys2 = ios.LinearIOSystem(linsys2, name='iosys2')

        # Connect systems in different ways and compare to StateSpace
        linsys_series = linsys2 * linsys1
        iosys_series = ios.InterconnectedSystem(
            [iosys1, iosys2],   # systems
            [[1, 0]],           # interconnection (series)
            0,                  # input = first system
            1                   # output = second system
        )

        # Run a simulation and compare to linear response
        T, U = tsys.T, tsys.U
        X0 = np.concatenate((tsys.X0, tsys.X0))
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        lti_t, lti_y = ct.forced_response(linsys_series, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

        # Connect systems with different timebases
        linsys2c = tsys.siso_linsys
        linsys2c.dt = 0         # Reset the timebase
        iosys2c = ios.LinearIOSystem(linsys2c)
        iosys_series = ios.InterconnectedSystem(
            [iosys1, iosys2c],   # systems
            [[1, 0]],          # interconnection (series)
            0,                  # input = first system
            1                   # output = second system
        )
        assert ct.isctime(iosys_series, strict=True)
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        lti_t, lti_y = ct.forced_response(linsys_series, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

        # Feedback interconnection
        linsys_feedback = ct.feedback(linsys1, linsys2)
        iosys_feedback = ios.InterconnectedSystem(
            [iosys1, iosys2],   # systems
            [[1, 0],            # input of sys2 = output of sys1
             [0, (1, 0, -1)]],  # input of sys1 = -output of sys2
            0,                  # input = first system
            0                   # output = first system
        )
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_feedback, T, U, X0, return_x=True)
        lti_t, lti_y = ct.forced_response(linsys_feedback, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    @pytest.mark.parametrize(
        "connections, inplist, outlist",
        [pytest.param([[(1, 0), (0, 0, 1)]], [[(0, 0, 1)]], [[(1, 0, 1)]],
                      id="full, raw tuple"),
         pytest.param([[(1, 0), (0, 0, -1)]], [[(0, 0)]], [[(1, 0, -1)]],
                      id="full, raw tuple, canceling gains"),
         pytest.param([[(1, 0), (0, 0)]], [[(0, 0)]], [[(1, 0)]],
                      id="full, raw tuple, no gain"),
         pytest.param([[(1, 0), (0, 0)]], [(0, 0)], [(1, 0)],
                      id="full, raw tuple, no gain, no outer list"),
         pytest.param([['sys2.u[0]', 'sys1.y[0]']], ['sys1.u[0]'],
                      ['sys2.y[0]'], id="named, full"),
         pytest.param([['sys2.u[0]', '-sys1.y[0]']], ['sys1.u[0]'],
                      ['-sys2.y[0]'], id="named, full, caneling gains"),
         pytest.param([['sys2.u[0]', 'sys1.y[0]']], 'sys1.u[0]', 'sys2.y[0]',
                      id="named, full, no list"),
         pytest.param([['sys2.u[0]', ('sys1', 'y[0]')]], [(0, 0)], [(1,)],
                      id="mixed"),
         pytest.param([[1, 0]], 0, 1, id="minimal")])
    def test_connect_spec_variants(self, tsys, connections, inplist, outlist):
        # Define a couple of (linear) systems to interconnection
        linsys1 = tsys.siso_linsys
        iosys1 = ios.LinearIOSystem(linsys1, name="sys1")
        linsys2 = tsys.siso_linsys
        iosys2 = ios.LinearIOSystem(linsys2, name="sys2")

        # Simple series connection
        linsys_series = linsys2 * linsys1

        # Create a simulation run to compare against
        T, U = tsys.T, tsys.U
        X0 = np.concatenate((tsys.X0, tsys.X0))
        lti_t, lti_y, lti_x = ct.forced_response(
            linsys_series, T, U, X0, return_x=True)

        # Create the input/output system with different parameter variations
        iosys_series = ios.InterconnectedSystem(
            [iosys1, iosys2], connections, inplist, outlist)
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y, atol=0.002, rtol=0.)

    @pytest.mark.parametrize(
        "connections, inplist, outlist",
        [pytest.param([['sys2.u[0]', 'sys1.y[0]']],
                      [[('sys1', 'u[0]'), ('sys1', 'u[0]')]],
                      [('sys2', 'y[0]', 0.5)], id="duplicated input"),
         pytest.param([['sys2.u[0]', ('sys1', 'y[0]', 0.5)],
                       ['sys2.u[0]', ('sys1', 'y[0]', 0.5)]],
                      'sys1.u[0]', 'sys2.y[0]', id="duplicated connection"),
         pytest.param([['sys2.u[0]', 'sys1.y[0]']], 'sys1.u[0]',
                      [[('sys2', 'y[0]', 0.5), ('sys2', 'y[0]', 0.5)]],
                      id="duplicated output")])
    def test_connect_spec_warnings(self, tsys, connections, inplist, outlist):
        # Define a couple of (linear) systems to interconnection
        linsys1 = tsys.siso_linsys
        iosys1 = ios.LinearIOSystem(linsys1, name="sys1")
        linsys2 = tsys.siso_linsys
        iosys2 = ios.LinearIOSystem(linsys2, name="sys2")

        # Simple series connection
        linsys_series = linsys2 * linsys1

        # Create a simulation run to compare against
        T, U = tsys.T, tsys.U
        X0 = np.concatenate((tsys.X0, tsys.X0))
        lti_t, lti_y, lti_x = ct.forced_response(
            linsys_series, T, U, X0, return_x=True)

        # Set up multiple gainst and make sure a warning is generated
        with pytest.warns(UserWarning, match="multiple.*Combining"):
            iosys_series = ios.InterconnectedSystem(
                [iosys1, iosys2], connections, inplist, outlist)
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y, atol=0.002, rtol=0.)

    def test_static_nonlinearity(self, tsys):
        # Linear dynamical system
        linsys = tsys.siso_linsys
        ioslin = ios.LinearIOSystem(linsys)

        # Nonlinear saturation
        sat = lambda u: u if abs(u) < 1 else np.sign(u)
        sat_output = lambda t, x, u, params: sat(u)
        nlsat =  ios.NonlinearIOSystem(None, sat_output, inputs=1, outputs=1)

        # Set up parameters for simulation
        T, U, X0 = tsys.T, 2 * tsys.U, tsys.X0
        Usat = np.vectorize(sat)(U)

        # Make sure saturation works properly by comparing linear system with
        # saturated input to nonlinear system with saturation composition
        lti_t, lti_y, lti_x = ct.forced_response(
            linsys, T, Usat, X0, return_x=True)
        ios_t, ios_y, ios_x = ios.input_output_response(
            ioslin * nlsat, T, U, X0, return_x=True)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_array_almost_equal(lti_y, ios_y, decimal=2)


    @pytest.mark.filterwarnings("ignore:Duplicate name::control.iosys")
    def test_algebraic_loop(self, tsys):
        # Create some linear and nonlinear systems to play with
        linsys = tsys.siso_linsys
        lnios = ios.LinearIOSystem(linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        nlios1 = nlios.copy(name='nlios1')
        nlios2 = nlios.copy(name='nlios2')

        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Single nonlinear system - no states
        ios_t, ios_y = ios.input_output_response(nlios, T, U)
        np.testing.assert_array_almost_equal(ios_y, U*U, decimal=3)

        # Composed nonlinear system (series)
        ios_t, ios_y = ios.input_output_response(nlios1 * nlios2, T, U)
        np.testing.assert_array_almost_equal(ios_y, U**4, decimal=3)

        # Composed nonlinear system (parallel)
        ios_t, ios_y = ios.input_output_response(nlios1 + nlios2, T, U)
        np.testing.assert_array_almost_equal(ios_y, 2*U**2, decimal=3)

        # Nonlinear system composed with LTI system (series) -- with states
        ios_t, ios_y = ios.input_output_response(
            nlios * lnios * nlios, T, U, X0)
        lti_t, lti_y = ct.forced_response(linsys, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, lti_y*lti_y, decimal=3)

        # Nonlinear system in feeback loop with LTI system
        iosys = ios.InterconnectedSystem(
            [lnios, nlios],         # linear system w/ nonlinear feedback
            [[1],                   # feedback interconnection (sig to 0)
             [0, (1, 0, -1)]],
            0,                      # input to linear system
            0                       # output from linear system
        )
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        # No easy way to test the result

        # Algebraic loop from static nonlinear system in feedback
        # (error will be due to no states)
        iosys = ios.InterconnectedSystem(
            [nlios1, nlios2],       # two copies of a static nonlinear system
            [[0, 1],                # feedback interconnection
             [1, (0, 0, -1)]],
            0, 0
        )
        args = (iosys, T, U)
        with pytest.raises(RuntimeError):
            ios.input_output_response(*args)

        # Algebraic loop due to feedthrough term
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[1]])
        lnios = ios.LinearIOSystem(linsys)
        iosys = ios.InterconnectedSystem(
            [nlios, lnios],         # linear system w/ nonlinear feedback
            [[0, 1],                # feedback interconnection
             [1, (0, 0, -1)]],
            0, 0
        )
        args = (iosys, T, U, X0)
        # ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        with pytest.raises(RuntimeError):
            ios.input_output_response(*args)

    def test_summer(self, tsys):
        # Construct a MIMO system for testing
        linsys = tsys.mimo_linsys1
        linio1 = ios.LinearIOSystem(linsys, name='linio1')
        linio2 = ios.LinearIOSystem(linsys, name='linio2')

        linsys_parallel = linsys + linsys
        iosys_parallel = linio1 + linio2

        # Set up parameters for simulation
        T = tsys.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        lin_t, lin_y = ct.forced_response(linsys_parallel, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_parallel, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_rmul(self, tsys):
        # Test right multiplication
        # TODO: replace with better tests when conversions are implemented

        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Linear system with input and output nonlinearities
        # Also creates a nested interconnected system
        ioslin = ios.LinearIOSystem(tsys.siso_linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        sys1 = nlios * ioslin
        sys2 = ios.InputOutputSystem.__rmul__(nlios, sys1)

        # Make sure we got the right thing (via simulation comparison)
        ios_t, ios_y = ios.input_output_response(sys2, T, U, X0)
        lti_t, lti_y = ct.forced_response(ioslin, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, lti_y*lti_y, decimal=3)

    def test_neg(self, tsys):
        """Test negation of a system"""

        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Static nonlinear system
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        ios_t, ios_y = ios.input_output_response(-nlios, T, U)
        np.testing.assert_array_almost_equal(ios_y, -U*U, decimal=3)

        # Linear system with input nonlinearity
        # Also creates a nested interconnected system
        ioslin = ios.LinearIOSystem(tsys.siso_linsys)
        sys = (ioslin) * (-nlios)

        # Make sure we got the right thing (via simulation comparison)
        ios_t, ios_y = ios.input_output_response(sys, T, U, X0)
        lti_t, lti_y = ct.forced_response(ioslin, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, -lti_y, decimal=3)

    def test_feedback(self, tsys):
        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Linear system with constant feedback (via "nonlinear" mapping)
        ioslin = ios.LinearIOSystem(tsys.siso_linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u, inputs=1, outputs=1)
        iosys = ct.feedback(ioslin, nlios)
        linsys = ct.feedback(tsys.siso_linsys, 1)

        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        lti_t, lti_y = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_y, lti_y,atol=0.002,rtol=0.)

    def test_bdalg_functions(self, tsys):
        """Test block diagram functions algebra on I/O systems"""
        # Set up parameters for simulation
        T = tsys.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        # Set up systems to be composed
        linsys1 = tsys.mimo_linsys1
        linio1 = ios.LinearIOSystem(linsys1)
        linsys2 = tsys.mimo_linsys2
        linio2 = ios.LinearIOSystem(linsys2)

        # Series interconnection
        linsys_series = ct.series(linsys1, linsys2)
        iosys_series = ct.series(linio1, linio2)
        lin_t, lin_y = ct.forced_response(linsys_series, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_series, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Make sure that systems don't commute
        linsys_series = ct.series(linsys2, linsys1)
        lin_t, lin_y = ct.forced_response(linsys_series, T, U, X0)
        assert not (np.abs(lin_y - ios_y) < 1e-3).all()

        # Parallel interconnection
        linsys_parallel = ct.parallel(linsys1, linsys2)
        iosys_parallel = ct.parallel(linio1, linio2)
        lin_t, lin_y = ct.forced_response(linsys_parallel, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_parallel, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Negation
        linsys_negate = ct.negate(linsys1)
        iosys_negate = ct.negate(linio1)
        lin_t, lin_y = ct.forced_response(linsys_negate, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_negate, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Feedback interconnection
        linsys_feedback = ct.feedback(linsys1, linsys2)
        iosys_feedback = ct.feedback(linio1, linio2)
        lin_t, lin_y = ct.forced_response(linsys_feedback, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_feedback, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_algebraic_functions(self, tsys):
        """Test algebraic operations on I/O systems"""
        # Set up parameters for simulation
        T = tsys.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        # Set up systems to be composed
        linsys1 = tsys.mimo_linsys1
        linio1 = ios.LinearIOSystem(linsys1)
        linsys2 = tsys.mimo_linsys2
        linio2 = ios.LinearIOSystem(linsys2)

        # Multiplication
        linsys_mul = linsys2 * linsys1
        iosys_mul = linio2 * linio1
        lin_t, lin_y = ct.forced_response(linsys_mul, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_mul, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Make sure that systems don't commute
        linsys_mul = linsys1 * linsys2
        lin_t, lin_y = ct.forced_response(linsys_mul, T, U, X0)
        assert not (np.abs(lin_y - ios_y) < 1e-3).all()

        # Addition
        linsys_add = linsys1 + linsys2
        iosys_add = linio1 + linio2
        lin_t, lin_y = ct.forced_response(linsys_add, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_add, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Subtraction
        linsys_sub = linsys1 - linsys2
        iosys_sub = linio1 - linio2
        lin_t, lin_y = ct.forced_response(linsys_sub, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_sub, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Make sure that systems don't commute
        linsys_sub = linsys2 - linsys1
        lin_t, lin_y = ct.forced_response(linsys_sub, T, U, X0)
        assert not (np.abs(lin_y - ios_y) < 1e-3).all()

        # Negation
        linsys_negate = -linsys1
        iosys_negate = -linio1
        lin_t, lin_y = ct.forced_response(linsys_negate, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_negate, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_nonsquare_bdalg(self, tsys):
        # Set up parameters for simulation
        T = tsys.T
        U2 = [np.sin(T), np.cos(T)]
        U3 = [np.sin(T), np.cos(T), T]
        X0 = 0

        # Set up systems to be composed
        linsys_2i3o = ct.StateSpace(
            [[-1, 1, 0], [0, -2, 0], [0, 0, -3]], [[1, 0], [0, 1], [1, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.zeros((3, 2)))
        iosys_2i3o = ios.LinearIOSystem(linsys_2i3o)

        linsys_3i2o = ct.StateSpace(
            [[-1, 1, 0], [0, -2, 0], [0, 0, -3]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 1], [0, 1, -1]], np.zeros((2, 3)))
        iosys_3i2o = ios.LinearIOSystem(linsys_3i2o)

        # Multiplication
        linsys_multiply = linsys_3i2o * linsys_2i3o
        iosys_multiply = iosys_3i2o * iosys_2i3o
        lin_t, lin_y = ct.forced_response(linsys_multiply, T, U2, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U2, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        linsys_multiply = linsys_2i3o * linsys_3i2o
        iosys_multiply = iosys_2i3o * iosys_3i2o
        lin_t, lin_y = ct.forced_response(linsys_multiply, T, U3, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Right multiplication
        # TODO: add real tests once conversion from other types is supported
        iosys_multiply = ios.InputOutputSystem.__rmul__(iosys_3i2o, iosys_2i3o)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Feedback
        linsys_multiply = ct.feedback(linsys_3i2o, linsys_2i3o)
        iosys_multiply = iosys_3i2o.feedback(iosys_2i3o)
        lin_t, lin_y = ct.forced_response(linsys_multiply, T, U3, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Mismatch should generate exception
        args = (iosys_3i2o, iosys_3i2o)
        with pytest.raises(ValueError):
            ct.series(*args)

    def test_discrete(self, tsys):
        """Test discrete time functionality"""
        # Create some linear and nonlinear systems to play with
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]], True)
        lnios = ios.LinearIOSystem(linsys)

        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Simulate and compare to LTI output
        ios_t, ios_y = ios.input_output_response(lnios, T, U, X0)
        lin_t, lin_y = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_t, lin_t,atol=0.002,rtol=0.)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Test MIMO system, converted to discrete time
        linsys = ct.StateSpace(tsys.mimo_linsys1)
        linsys.dt = tsys.T[1] - tsys.T[0]
        lnios = ios.LinearIOSystem(linsys)

        # Set up parameters for simulation
        T = tsys.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        # Simulate and compare to LTI output
        ios_t, ios_y = ios.input_output_response(lnios, T, U, X0)
        lin_t, lin_y = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_t, lin_t,atol=0.002,rtol=0.)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_discrete_iosys(self, tsys):
        """Create a discrete time system from scratch"""
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]], True)

        # Create nonlinear version of the same system
        def nlsys_update(t, x, u, params):
            A, B = params['A'], params['B']
            return A @ x + B @ u
        def nlsys_output(t, x, u, params):
            C = params['C']
            return C @ x
        nlsys = ct.NonlinearIOSystem(
            nlsys_update, nlsys_output, inputs=1, outputs=1, states=2, dt=True)

        # Set up parameters for simulation
        T, U, X0 = tsys.T, tsys.U, tsys.X0

        # Simulate and compare to LTI output
        ios_t, ios_y = ios.input_output_response(
            nlsys, T, U, X0,
            params={'A': linsys.A, 'B': linsys.B, 'C': linsys.C})
        lin_t, lin_y = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_t, lin_t,atol=0.002,rtol=0.)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_find_eqpts_dfan(self, tsys):
        """Test find_eqpt function on dfan example"""
        # Simple equilibrium point with no inputs
        nlsys = ios.NonlinearIOSystem(predprey)
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [1.6, 1.2], None, return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(xeq, [1.64705879, 1.17923874])
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((2,)))

        # Ducted fan dynamics with output = velocity
        nlsys = ios.NonlinearIOSystem(pvtol, lambda t, x, u, params: x[0:2])

        # Make sure the origin is a fixed point
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0, 4*9.8], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)))
        np.testing.assert_array_almost_equal(xeq, [0, 0, 0, 0])

        # Use a small lateral force to cause motion
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Equilibrium point with fixed output
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0.1, 0.1], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Specify outputs to constrain (replicate previous)
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], y0=[0.1, 0.1],
            iy = [0, 1], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Specify inputs to constrain (replicate previous), w/ no result
        xeq, ueq = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], y0=[0.1, 0.1], iu = [])
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Now solve the problem with the original PVTOL variables
        # Constrain the output angle and x velocity
        nlsys_full = ios.NonlinearIOSystem(pvtol_full, None)
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0.1, 0.1, 0, 0], iy = [2, 3],
            idx=[2, 3, 4, 5], ix=[0, 1], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[[2, 3]], [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-4:], np.zeros((4,)), decimal=5)

        # Same test as before, but now all constraints are in the state vector
        nlsys_full = ios.NonlinearIOSystem(pvtol_full, None)
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0.1, 0.1, 0, 0], [0.01, 4*9.8],
            idx=[2, 3, 4, 5], ix=[0, 1, 2, 3], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[[2, 3]], [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-4:], np.zeros((4,)), decimal=5)

        # Fix one input and vary the other
        nlsys_full = ios.NonlinearIOSystem(pvtol_full, None)
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0.1, 0.1, 0, 0], iy=[3], iu=[1],
            idx=[2, 3, 4, 5], ix=[0, 1], return_result=True)
        assert result.success
        np.testing.assert_almost_equal(ueq[1], 4*9.8, decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[[3]], [0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-4:], np.zeros((4,)), decimal=5)

        # PVTOL with output = y velocity
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0.1, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0, 0.1, 0, 0], iy=[3],
            dx0=[0.1, 0, 0, 0, 0, 0], idx=[1, 2, 3, 4, 5],
            ix=[0, 1], return_result=True)
        assert result.success
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[-3:], [0.1, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-5:], np.zeros((5,)), decimal=5)

        # Unobservable system
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[0, 0]], [[0]])
        lnios = ios.LinearIOSystem(linsys)

        # If result is returned, user has to check
        xeq, ueq, result = ios.find_eqpt(
            lnios, [0, 0], [0], y0=[1], return_result=True)
        assert not result.success

        # If result is not returned, find_eqpt should return None
        xeq, ueq = ios.find_eqpt(lnios, [0, 0], [0], y0=[1])
        assert xeq is None
        assert ueq is None

    def test_params(self, tsys):
        # Start with the default set of parameters
        ios_secord_default = ios.NonlinearIOSystem(
            secord_update, secord_output, inputs=1, outputs=1, states=2)
        lin_secord_default = ios.linearize(ios_secord_default, [0, 0], [0])
        w_default, v_default = np.linalg.eig(lin_secord_default.A)

        # New copy, with modified parameters
        ios_secord_update = ios.NonlinearIOSystem(
            secord_update, secord_output, inputs=1, outputs=1, states=2,
            params={'omega0':2, 'zeta':0})

        # Make sure the default parameters haven't changed
        lin_secord_check = ios.linearize(ios_secord_default, [0, 0], [0])
        w, v = np.linalg.eig(lin_secord_check.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort(w_default))

        # Make sure updated system parameters got set correctly
        lin_secord_update = ios.linearize(ios_secord_update, [0, 0], [0])
        w, v = np.linalg.eig(lin_secord_update.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([2j, -2j]))

        # Change the parameters of the default sys just for the linearization
        lin_secord_local = ios.linearize(ios_secord_default, [0, 0], [0],
                                          params={'zeta':0})
        w, v = np.linalg.eig(lin_secord_local.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([1j, -1j]))

        # Change the parameters of the updated sys just for the linearization
        lin_secord_local = ios.linearize(ios_secord_update, [0, 0], [0],
                                          params={'zeta':0, 'omega0':3})
        w, v = np.linalg.eig(lin_secord_local.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([3j, -3j]))

        # Make sure that changes propagate through interconnections
        ios_series_default_local = ios_secord_default * ios_secord_update
        lin_series_default_local = ios.linearize(
            ios_series_default_local, [0, 0, 0, 0], [0])
        w, v = np.linalg.eig(lin_series_default_local.A)
        np.testing.assert_array_almost_equal(
            np.sort(w), np.sort(np.concatenate((w_default, [2j, -2j]))))

        # Show that we can change the parameters at linearization
        lin_series_override = ios.linearize(
            ios_series_default_local, [0, 0, 0, 0], [0],
            params={'zeta':0, 'omega0':4})
        w, v = np.linalg.eig(lin_series_override.A)
        np.testing.assert_array_almost_equal(w, [4j, -4j, 4j, -4j])

        # Check for warning if we try to set params for LinearIOSystem
        linsys = tsys.siso_linsys
        iosys = ios.LinearIOSystem(linsys)
        T, U, X0 = tsys.T, tsys.U, tsys.X0
        lti_t, lti_y = ct.forced_response(linsys, T, U, X0)
        with pytest.warns(UserWarning, match="LinearIOSystem.*ignored"):
            ios_t, ios_y = ios.input_output_response(
                iosys, T, U, X0, params={'something':0})

        # Check to make sure results are OK
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    def test_named_signals(self, tsys):
        sys1 = ios.NonlinearIOSystem(
            updfcn = lambda t, x, u, params: np.array(
                tsys.mimo_linsys1.A @ np.reshape(x, (-1, 1)) \
                + tsys.mimo_linsys1.B @ np.reshape(u, (-1, 1))
            ).reshape(-1,),
            outfcn = lambda t, x, u, params: np.array(
                tsys.mimo_linsys1.C @ np.reshape(x, (-1, 1)) \
                + tsys.mimo_linsys1.D @ np.reshape(u, (-1, 1))
            ).reshape(-1,),
            inputs = ['u[0]', 'u[1]'],
            outputs = ['y[0]', 'y[1]'],
            states = tsys.mimo_linsys1.nstates,
            name = 'sys1')
        sys2 = ios.LinearIOSystem(tsys.mimo_linsys2,
            inputs = ['u[0]', 'u[1]'],
            outputs = ['y[0]', 'y[1]'],
            name = 'sys2')

        # Series interconnection (sys1 * sys2) using __mul__
        ios_mul = sys1 * sys2
        ss_series = tsys.mimo_linsys1 * tsys.mimo_linsys2
        lin_series = ct.linearize(ios_mul, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Series interconnection (sys1 * sys2) using series
        ios_series = ct.series(sys2, sys1)
        ss_series = ct.series(tsys.mimo_linsys2, tsys.mimo_linsys1)
        lin_series = ct.linearize(ios_series, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Series interconnection (sys1 * sys2) using named + mixed signals
        ios_connect = ios.InterconnectedSystem(
            [sys2, sys1],
            connections=[
                [('sys1', 'u[0]'), 'sys2.y[0]'],
                ['sys1.u[1]', 'sys2.y[1]']
            ],
            inplist=['sys2.u[0]', ('sys2', 1)],
            outlist=[(1, 'y[0]'), 'sys1.y[1]']
        )
        lin_series = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Try the same thing using the interconnect function
        # Since sys1 is nonlinear, we should get back the same result
        ios_connect = ios.interconnect(
            (sys2, sys1),
            connections=(
                [('sys1', 'u[0]'), 'sys2.y[0]'],
                ['sys1.u[1]', 'sys2.y[1]']
            ),
            inplist=['sys2.u[0]', ('sys2', 1)],
            outlist=[(1, 'y[0]'), 'sys1.y[1]']
        )
        lin_series = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Try the same thing using the interconnect function
        # Since sys1 is nonlinear, we should get back the same result
        # Note: use a tuple for connections to make sure it works
        ios_connect = ios.interconnect(
            (sys2, sys1),
            connections=(
                [('sys1', 'u[0]'), 'sys2.y[0]'],
                ['sys1.u[1]', 'sys2.y[1]']
            ),
            inplist=['sys2.u[0]', ('sys2', 1)],
            outlist=[(1, 'y[0]'), 'sys1.y[1]']
        )
        lin_series = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Make sure that we can use input signal names as system outputs
        ios_connect = ios.InterconnectedSystem(
            [sys1, sys2],
            connections=[
                ['sys2.u[0]', 'sys1.y[0]'], ['sys2.u[1]', 'sys1.y[1]'],
                ['sys1.u[0]', '-sys2.y[0]'], ['sys1.u[1]', '-sys2.y[1]']
            ],
            inplist=['sys1.u[0]', 'sys1.u[1]'],
            outlist=['sys2.u[0]', 'sys2.u[1]']  # = sys1.y[0], sys1.y[1]
        )
        ss_feedback = ct.feedback(tsys.mimo_linsys1, tsys.mimo_linsys2)
        lin_feedback = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_feedback.A, lin_feedback.A)
        np.testing.assert_array_almost_equal(ss_feedback.B, lin_feedback.B)
        np.testing.assert_array_almost_equal(ss_feedback.C, lin_feedback.C)
        np.testing.assert_array_almost_equal(ss_feedback.D, lin_feedback.D)

    @pytest.mark.usefixtures("editsdefaults")
    def test_sys_naming_convention(self, tsys):
        """Enforce generic system names 'sys[i]' to be present when systems are
        created without explicit names."""

        ct.config.use_legacy_defaults('0.8.4')  # changed delims in 0.9.0
        ct.config.use_numpy_matrix(False)       # np.matrix deprecated

        # Create a system with a known ID
        ct.namedio.NamedIOSystem._idCounter = 0
        sys = ct.ss(
            tsys.mimo_linsys1.A, tsys.mimo_linsys1.B,
            tsys.mimo_linsys1.C, tsys.mimo_linsys1.D)

        assert sys.name == "sys[0]"
        assert sys.copy().name == "copy of sys[0]"

        namedsys = ios.NonlinearIOSystem(
            updfcn=lambda t, x, u, params: x,
            outfcn=lambda t, x, u, params: u,
            inputs=('u[0]', 'u[1]'),
            outputs=('y[0]', 'y[1]'),
            states=tsys.mimo_linsys1.nstates,
            name='namedsys')
        unnamedsys1 = ct.NonlinearIOSystem(
            lambda t, x, u, params: x, inputs=2, outputs=2, states=2
        )
        unnamedsys2 = ct.NonlinearIOSystem(
            None, lambda t, x, u, params: u, inputs=2, outputs=2
        )
        assert unnamedsys2.name == "sys[2]"

        # Unnamed/unnamed connections
        uu_series = unnamedsys1 * unnamedsys2
        uu_parallel = unnamedsys1 + unnamedsys2
        u_neg = - unnamedsys1
        uu_feedback = unnamedsys2.feedback(unnamedsys1)
        uu_dup = unnamedsys1 * unnamedsys1.copy()
        uu_hierarchical = uu_series * unnamedsys1

        assert uu_series.name == "sys[3]"
        assert uu_parallel.name == "sys[4]"
        assert u_neg.name == "sys[5]"
        assert uu_feedback.name == "sys[6]"
        assert uu_dup.name == "sys[7]"
        assert uu_hierarchical.name == "sys[8]"

        # Unnamed/named connections
        un_series = unnamedsys1 * namedsys
        un_parallel = unnamedsys1 + namedsys
        un_feedback = unnamedsys2.feedback(namedsys)
        un_dup = unnamedsys1 * namedsys.copy()
        un_hierarchical = uu_series * unnamedsys1

        assert un_series.name == "sys[9]"
        assert un_parallel.name == "sys[10]"
        assert un_feedback.name == "sys[11]"
        assert un_dup.name == "sys[12]"
        assert un_hierarchical.name == "sys[13]"

        # Same system conflict
        with pytest.warns(UserWarning):
            namedsys * namedsys

    @pytest.mark.usefixtures("editsdefaults")
    def test_signals_naming_convention_0_8_4(self, tsys):
        """Enforce generic names to be present when systems are created
        without explicit signal names:
        input: 'u[i]'
        state: 'x[i]'
        output: 'y[i]'
        """

        ct.config.use_legacy_defaults('0.8.4')  # changed delims in 0.9.0
        ct.config.use_numpy_matrix(False)       # np.matrix deprecated

        # Create a system with a known ID
        ct.namedio.NamedIOSystem._idCounter = 0
        sys = ct.ss(
            tsys.mimo_linsys1.A, tsys.mimo_linsys1.B,
            tsys.mimo_linsys1.C, tsys.mimo_linsys1.D)

        for statename in ["x[0]", "x[1]"]:
            assert statename in sys.state_index
        for inputname in ["u[0]", "u[1]"]:
            assert inputname in sys.input_index
        for outputname in ["y[0]", "y[1]"]:
            assert outputname in sys.output_index
        assert len(sys.state_index) == sys.nstates
        assert len(sys.input_index) == sys.ninputs
        assert len(sys.output_index) == sys.noutputs

        namedsys = ios.NonlinearIOSystem(
            updfcn=lambda t, x, u, params: x,
            outfcn=lambda t, x, u, params: u,
            inputs=('u0'),
            outputs=('y0'),
            states=('x0'),
            name='namedsys')
        unnamedsys = ct.NonlinearIOSystem(
            lambda t, x, u, params: x, inputs=1, outputs=1, states=1
        )
        assert 'u0' in namedsys.input_index
        assert 'y0' in namedsys.output_index
        assert 'x0' in namedsys.state_index

        # Unnamed/named connections
        un_series = unnamedsys * namedsys
        un_parallel = unnamedsys + namedsys
        un_feedback = unnamedsys.feedback(namedsys)
        un_dup = unnamedsys * namedsys.copy()
        un_hierarchical = un_series*unnamedsys
        u_neg = - unnamedsys

        assert "sys[1].x[0]" in un_series.state_index
        assert "namedsys.x0" in un_series.state_index
        assert "sys[1].x[0]" in un_parallel.state_index
        assert "namedsys.x0" in un_series.state_index
        assert "sys[1].x[0]" in un_feedback.state_index
        assert "namedsys.x0" in un_feedback.state_index
        assert "sys[1].x[0]" in un_dup.state_index
        assert "copy of namedsys.x0" in un_dup.state_index
        assert "sys[1].x[0]" in un_hierarchical.state_index
        assert "sys[2].sys[1].x[0]" in un_hierarchical.state_index
        assert "sys[1].x[0]" in u_neg.state_index

        # Same system conflict
        with pytest.warns(UserWarning):
            same_name_series = namedsys * namedsys
            assert "namedsys.x0" in same_name_series.state_index
            assert "copy of namedsys.x0" in same_name_series.state_index

    def test_named_signals_linearize_inconsistent(self, tsys):
        """Make sure that providing inputs or outputs not consistent with
           updfcn or outfcn fail
        """

        def updfcn(t, x, u, params):
            """2 inputs, 2 states"""
            return np.array(
                tsys.mimo_linsys1.A @ np.reshape(x, (-1, 1))
                + tsys.mimo_linsys1.B @ np.reshape(u, (-1, 1))
                ).reshape(-1,)

        def outfcn(t, x, u, params):
            """2 states, 2 outputs"""
            return np.array(
                    tsys.mimo_linsys1.C * np.reshape(x, (-1, 1))
                    + tsys.mimo_linsys1.D * np.reshape(u, (-1, 1))
                ).reshape(-1,)

        for inputs, outputs in [
                (('u[0]'), ('y[0]', 'y[1]')),  # not enough u
                (('u[0]', 'u[1]', 'u[toomuch]'), ('y[0]', 'y[1]')),
                (('u[0]', 'u[1]'), ('y[0]')),  # not enough y
                (('u[0]', 'u[1]'), ('y[0]', 'y[1]', 'y[toomuch]'))]:
            sys1 = ios.NonlinearIOSystem(updfcn=updfcn,
                                         outfcn=outfcn,
                                         inputs=inputs,
                                         outputs=outputs,
                                         states=tsys.mimo_linsys1.nstates,
                                         name='sys1')
            with pytest.raises(ValueError):
                sys1.linearize([0, 0], [0, 0])

        sys2 = ios.NonlinearIOSystem(updfcn=updfcn,
                                     outfcn=outfcn,
                                     inputs=('u[0]', 'u[1]'),
                                     outputs=('y[0]', 'y[1]'),
                                     states=tsys.mimo_linsys1.nstates,
                                     name='sys1')
        for x0, u0 in [([0], [0, 0]),
                       ([0, 0, 0], [0, 0]),
                       ([0, 0], [0]),
                       ([0, 0], [0, 0, 0])]:
            with pytest.raises(ValueError):
                sys2.linearize(x0, u0)

    def test_linearize_concatenation(self, kincar):
        # Create a simple nonlinear system to check (kinematic car)
        iosys = kincar
        linearized = iosys.linearize([0, np.array([0, 0])], [0, 0])
        np.testing.assert_array_almost_equal(linearized.A, np.zeros((3,3)))
        np.testing.assert_array_almost_equal(
            linearized.B, [[1, 0], [0, 0], [0, 1]])
        np.testing.assert_array_almost_equal(
            linearized.C, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_almost_equal(linearized.D, np.zeros((2,2)))

    def test_lineariosys_statespace(self, tsys):
        """Make sure that a LinearIOSystem is also a StateSpace object"""
        iosys_siso = ct.LinearIOSystem(tsys.siso_linsys, name='siso')
        iosys_siso2 = ct.LinearIOSystem(tsys.siso_linsys, name='siso2')
        assert isinstance(iosys_siso, ct.StateSpace)

        # Make sure that state space functions work for LinearIOSystems
        np.testing.assert_allclose(
            iosys_siso.poles(), tsys.siso_linsys.poles())
        omega = np.logspace(.1, 10, 100)
        mag_io, phase_io, omega_io = iosys_siso.frequency_response(omega)
        mag_ss, phase_ss, omega_ss = tsys.siso_linsys.frequency_response(omega)
        np.testing.assert_allclose(mag_io, mag_ss)
        np.testing.assert_allclose(phase_io, phase_ss)
        np.testing.assert_allclose(omega_io, omega_ss)

        # LinearIOSystem methods should override StateSpace methods
        io_mul = iosys_siso * iosys_siso2
        assert isinstance(io_mul, ct.InputOutputSystem)

        # But also retain linear structure
        assert isinstance(io_mul, ct.StateSpace)

        # And make sure the systems match
        ss_series = tsys.siso_linsys * tsys.siso_linsys
        np.testing.assert_allclose(io_mul.A, ss_series.A)
        np.testing.assert_allclose(io_mul.B, ss_series.B)
        np.testing.assert_allclose(io_mul.C, ss_series.C)
        np.testing.assert_allclose(io_mul.D, ss_series.D)

        # Make sure that series does the same thing
        io_series = ct.series(iosys_siso, iosys_siso2)
        assert isinstance(io_series, ct.InputOutputSystem)
        assert isinstance(io_series, ct.StateSpace)
        np.testing.assert_allclose(io_series.A, ss_series.A)
        np.testing.assert_allclose(io_series.B, ss_series.B)
        np.testing.assert_allclose(io_series.C, ss_series.C)
        np.testing.assert_allclose(io_series.D, ss_series.D)

        # Test out feedback as well
        io_feedback = ct.feedback(iosys_siso, iosys_siso2)
        assert isinstance(io_series, ct.InputOutputSystem)

        # But also retain linear structure
        assert isinstance(io_series, ct.StateSpace)

        # And make sure the systems match
        ss_feedback = ct.feedback(tsys.siso_linsys, tsys.siso_linsys)
        np.testing.assert_allclose(io_feedback.A, ss_feedback.A)
        np.testing.assert_allclose(io_feedback.B, ss_feedback.B)
        np.testing.assert_allclose(io_feedback.C, ss_feedback.C)
        np.testing.assert_allclose(io_feedback.D, ss_feedback.D)

        # Make sure series interconnections are done in the right order
        ss_sys1 = ct.rss(2, 3, 2)
        io_sys1 = ct.ss2io(ss_sys1)
        ss_sys2 = ct.rss(2, 2, 3)
        io_sys2 = ct.ss2io(ss_sys2)
        io_series = io_sys2 * io_sys1
        assert io_series.ninputs == 2
        assert io_series.noutputs == 2
        assert io_series.nstates == 4

        # While we are at it, check that the state space matrices match
        ss_series = ss_sys2 * ss_sys1
        np.testing.assert_allclose(io_series.A, ss_series.A)
        np.testing.assert_allclose(io_series.B, ss_series.B)
        np.testing.assert_allclose(io_series.C, ss_series.C)
        np.testing.assert_allclose(io_series.D, ss_series.D)

    @pytest.mark.parametrize(
        "Pout, Pin, C, op, PCout, PCin", [
            (2, 2, 'rss', ct.LinearIOSystem.__mul__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__mul__, 2, 2),
            (2, 3, 2, ct.LinearIOSystem.__mul__, 2, 3),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__mul__, 2, 2),
            (2, 2, 'rss', ct.LinearIOSystem.__rmul__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__rmul__, 2, 2),
            (2, 3, 2, ct.LinearIOSystem.__rmul__, 2, 3),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__rmul__, 2, 2),
            (2, 2, 'rss', ct.LinearIOSystem.__add__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__add__, 2, 2),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__add__, 2, 2),
            (2, 2, 'rss', ct.LinearIOSystem.__radd__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__radd__, 2, 2),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__radd__, 2, 2),
            (2, 2, 'rss', ct.LinearIOSystem.__sub__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__sub__, 2, 2),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__sub__, 2, 2),
            (2, 2, 'rss', ct.LinearIOSystem.__rsub__, 2, 2),
            (2, 2, 2, ct.LinearIOSystem.__rsub__, 2, 2),
            (2, 2, np.random.rand(2, 2), ct.LinearIOSystem.__rsub__, 2, 2),

        ])
    def test_operand_conversion(self, Pout, Pin, C, op, PCout, PCin):
        P = ct.LinearIOSystem(
            ct.rss(2, Pout, Pin, strictly_proper=True), name='P')
        if isinstance(C, str) and C == 'rss':
            # Need to generate inside class to avoid matrix deprecation error
            C = ct.rss(2, 2, 2)
        PC = op(P, C)
        assert isinstance(PC, ct.LinearIOSystem)
        assert isinstance(PC, ct.StateSpace)
        assert PC.noutputs == PCout
        assert PC.ninputs == PCin

    @pytest.mark.parametrize(
        "Pout, Pin, C, op", [
            (2, 2, 'rss32', ct.LinearIOSystem.__mul__),
            (2, 2, 'rss23', ct.LinearIOSystem.__rmul__),
            (2, 2, 'rss32', ct.LinearIOSystem.__add__),
            (2, 2, 'rss23', ct.LinearIOSystem.__radd__),
            (2, 3, 2, ct.LinearIOSystem.__add__),
            (2, 3, 2, ct.LinearIOSystem.__radd__),
            (2, 2, 'rss32', ct.LinearIOSystem.__sub__),
            (2, 2, 'rss23', ct.LinearIOSystem.__rsub__),
            (2, 3, 2, ct.LinearIOSystem.__sub__),
            (2, 3, 2, ct.LinearIOSystem.__rsub__),
        ])
    def test_operand_incompatible(self, Pout, Pin, C, op):
        P = ct.LinearIOSystem(
            ct.rss(2, Pout, Pin, strictly_proper=True), name='P')
        if isinstance(C, str) and C == 'rss32':
            C = ct.rss(2, 3, 2)
        elif isinstance(C, str) and C == 'rss23':
            C = ct.rss(2, 2, 3)
        with pytest.raises(ValueError, match="incompatible"):
            PC = op(P, C)

    @pytest.mark.parametrize(
        "C, op", [
            (None, ct.LinearIOSystem.__mul__),
            (None, ct.LinearIOSystem.__rmul__),
            (None, ct.LinearIOSystem.__add__),
            (None, ct.LinearIOSystem.__radd__),
            (None, ct.LinearIOSystem.__sub__),
            (None, ct.LinearIOSystem.__rsub__),
        ])
    def test_operand_badtype(self, C, op):
        P = ct.LinearIOSystem(
            ct.rss(2, 2, 2, strictly_proper=True), name='P')
        with pytest.raises(TypeError, match="Unknown"):
            op(P, C)

    def test_neg_badsize(self):
        # Create a system of unspecified size
        sys = ct.InputOutputSystem()
        with pytest.raises(ValueError, match="Can't determine"):
            -sys

    def test_bad_signal_list(self):
        # Create a ystem with a bad signal list
        with pytest.raises(TypeError, match="Can't parse"):
            ct.InputOutputSystem(inputs=[1, 2, 3])

    def test_docstring_example(self):
        P = ct.LinearIOSystem(
            ct.rss(2, 2, 2, strictly_proper=True), name='P')
        C = ct.LinearIOSystem(ct.rss(2, 2, 2), name='C')
        S = ct.InterconnectedSystem(
            [C, P],
            connections = [
              ['P.u[0]', 'C.y[0]'], ['P.u[1]', 'C.y[1]'],
              ['C.u[0]', '-P.y[0]'], ['C.u[1]', '-P.y[1]']],
            inplist = ['C.u[0]', 'C.u[1]'],
            outlist = ['P.y[0]', 'P.y[1]'],
        )
        ss_P = ct.StateSpace(P.linearize(0, 0))
        ss_C = ct.StateSpace(C.linearize(0, 0))
        ss_eye = ct.StateSpace(
            [], np.zeros((0, 2)), np.zeros((2, 0)), np.eye(2))
        ss_S = ct.feedback(ss_P * ss_C, ss_eye)
        io_S = S.linearize(0, 0)
        np.testing.assert_array_almost_equal(io_S.A, ss_S.A)
        np.testing.assert_array_almost_equal(io_S.B, ss_S.B)
        np.testing.assert_array_almost_equal(io_S.C, ss_S.C)
        np.testing.assert_array_almost_equal(io_S.D, ss_S.D)

    @pytest.mark.usefixtures("editsdefaults")
    def test_duplicates(self, tsys):
        nlios = ios.NonlinearIOSystem(lambda t, x, u, params: x,
                                      lambda t, x, u, params: u * u,
                                      inputs=1, outputs=1, states=1,
                                      name="sys")

        # Duplicate objects
        with pytest.warns(UserWarning, match="duplicate object"):
            ios_series = nlios * nlios

        # Nonduplicate objects
        ct.config.use_legacy_defaults('0.8.4')  # changed delims in 0.9.0
        ct.config.use_numpy_matrix(False)       # np.matrix deprecated
        nlios1 = nlios.copy()
        nlios2 = nlios.copy()
        with pytest.warns(UserWarning, match="duplicate name"):
            ios_series = nlios1 * nlios2
            assert "copy of sys_1.x[0]" in ios_series.state_index.keys()
            assert "copy of sys.x[0]" in ios_series.state_index.keys()

        # Duplicate names
        iosys_siso = ct.LinearIOSystem(tsys.siso_linsys)
        nlios1 = ios.NonlinearIOSystem(None,
                                       lambda t, x, u, params: u * u,
                                       inputs=1, outputs=1, name="sys")
        nlios2 = ios.NonlinearIOSystem(None,
                                       lambda t, x, u, params: u * u,
                                       inputs=1, outputs=1, name="sys")

        with pytest.warns(UserWarning, match="duplicate name"):
            ct.InterconnectedSystem([nlios1, iosys_siso, nlios2],
                                    inputs=0, outputs=0, states=0)

        # Same system, different names => everything should be OK
        nlios1 = ios.NonlinearIOSystem(None,
                                       lambda t, x, u, params:  u * u,
                                       inputs=1, outputs=1, name="nlios1")
        nlios2 = ios.NonlinearIOSystem(None,
                                       lambda t, x, u, params: u * u,
                                       inputs=1, outputs=1, name="nlios2")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ct.InterconnectedSystem([nlios1, iosys_siso, nlios2],
                                    inputs=0, outputs=0, states=0)


def test_linear_interconnection():
    ss_sys1 = ct.rss(2, 2, 2, strictly_proper=True)
    ss_sys2 = ct.rss(2, 2, 2)
    io_sys1 = ios.LinearIOSystem(
        ss_sys1, inputs = ('u[0]', 'u[1]'),
        outputs = ('y[0]', 'y[1]'), name = 'sys1')
    io_sys2 = ios.LinearIOSystem(
        ss_sys2, inputs = ('u[0]', 'u[1]'),
        outputs = ('y[0]', 'y[1]'), name = 'sys2')
    nl_sys2 = ios.NonlinearIOSystem(
        lambda t, x, u, params: np.array(
            ss_sys2.A @ np.reshape(x, (-1, 1)) \
            + ss_sys2.B @ np.reshape(u, (-1, 1))
            ).reshape((-1,)),
        lambda t, x, u, params: np.array(
            ss_sys2.C @ np.reshape(x, (-1, 1)) \
            + ss_sys2.D @ np.reshape(u, (-1, 1))
            ).reshape((-1,)),
        states = 2,
        inputs = ('u[0]', 'u[1]'),
        outputs = ('y[0]', 'y[1]'),
        name = 'sys2')
    tf_siso = ct.tf(1, [0.1, 1])
    ss_siso = ct.ss(1, 2, 1, 1)
    nl_siso = ios.NonlinearIOSystem(
        lambda t, x, u, params: x*x,
        lambda t, x, u, params: u*x, states=1, inputs=1, outputs=1)

    # Create a "regular" InterconnectedSystem
    nl_connect = ios.interconnect(
        (io_sys1, nl_sys2),
        connections=[
            ['sys1.u[1]', 'sys2.y[0]'],
            ['sys2.u[0]', 'sys1.y[1]']
        ],
        inplist=[
            ['sys1.u[0]', 'sys1.u[1]'],
            ['sys2.u[1]']],
        outlist=[
            ['sys1.y[0]', '-sys2.y[0]'],
            ['sys2.y[1]'],
            ['sys2.u[1]']])
    assert isinstance(nl_connect, ios.InterconnectedSystem)
    assert not isinstance(nl_connect, ios.LinearICSystem)

    # Now take its linearization
    ss_connect = nl_connect.linearize(0, 0)
    assert isinstance(ss_connect, ios.LinearIOSystem)

    io_connect = ios.interconnect(
        (io_sys1, io_sys2),
        connections=[
            ['sys1.u[1]', 'sys2.y[0]'],
            ['sys2.u[0]', 'sys1.y[1]']
        ],
        inplist=[
            ['sys1.u[0]', 'sys1.u[1]'],
            ['sys2.u[1]']],
        outlist=[
            ['sys1.y[0]', '-sys2.y[0]'],
            ['sys2.y[1]'],
            ['sys2.u[1]']])
    assert isinstance(io_connect, ios.InterconnectedSystem)
    assert isinstance(io_connect, ios.LinearICSystem)
    assert isinstance(io_connect, ios.LinearIOSystem)
    assert isinstance(io_connect, ct.StateSpace)

    # Finally compare the linearization with the linear system
    np.testing.assert_array_almost_equal(io_connect.A, ss_connect.A)
    np.testing.assert_array_almost_equal(io_connect.B, ss_connect.B)
    np.testing.assert_array_almost_equal(io_connect.C, ss_connect.C)
    np.testing.assert_array_almost_equal(io_connect.D, ss_connect.D)

    # make sure interconnections of linear systems are linear and
    # if a nonlinear system is included then system is nonlinear
    assert isinstance(ss_siso*ss_siso, ios.LinearIOSystem)
    assert isinstance(tf_siso*ss_siso, ios.LinearIOSystem)
    assert isinstance(ss_siso*tf_siso, ios.LinearIOSystem)
    assert ~isinstance(ss_siso*nl_siso, ios.LinearIOSystem)
    assert ~isinstance(nl_siso*ss_siso, ios.LinearIOSystem)
    assert ~isinstance(nl_siso*nl_siso, ios.LinearIOSystem)
    assert ~isinstance(tf_siso*nl_siso, ios.LinearIOSystem)
    assert ~isinstance(nl_siso*tf_siso, ios.LinearIOSystem)
    assert ~isinstance(nl_siso*nl_siso, ios.LinearIOSystem)


def predprey(t, x, u, params={}):
    """Predator prey dynamics"""
    r = params.get('r', 2)
    d = params.get('d', 0.7)
    b = params.get('b', 0.3)
    k = params.get('k', 10)
    a = params.get('a', 8)
    c = params.get('c', 4)

    # Dynamics for the system
    dx0 = r * x[0] * (1 - x[0]/k) - a * x[1] * x[0]/(c + x[0])
    dx1 = b * a * x[1] * x[0] / (c + x[0]) - d * x[1]

    return np.array([dx0, dx1])


def pvtol(t, x, u, params={}):
    """Reduced planar vertical takeoff and landing dynamics"""
    from math import cos, sin
    m = params.get('m', 4.)      # kg, system mass
    J = params.get('J', 0.0475)  # kg m^2, system inertia
    r = params.get('r', 0.25)    # m, thrust offset
    g = params.get('g', 9.8)     # m/s, gravitational constant
    c = params.get('c', 0.05)    # N s/m, rotational damping
    l = params.get('c', 0.1)     # m, pivot location
    return np.array([
        x[3],
        -c/m * x[1] + 1/m * cos(x[0]) * u[0] - 1/m * sin(x[0]) * u[1],
        -g - c/m * x[2] + 1/m * sin(x[0]) * u[0] + 1/m * cos(x[0]) * u[1],
        -l/J * sin(x[0]) + r/J * u[0]
    ])


def pvtol_full(t, x, u, params={}):
    from math import cos, sin
    m = params.get('m', 4.)      # kg, system mass
    J = params.get('J', 0.0475)  # kg m^2, system inertia
    r = params.get('r', 0.25)    # m, thrust offset
    g = params.get('g', 9.8)     # m/s, gravitational constant
    c = params.get('c', 0.05)    # N s/m, rotational damping
    l = params.get('c', 0.1)     # m, pivot location
    return np.array([
        x[3], x[4], x[5],
        -c/m * x[3] + 1/m * cos(x[2]) * u[0] - 1/m * sin(x[2]) * u[1],
        -g - c/m * x[4] + 1/m * sin(x[2]) * u[0] + 1/m * cos(x[2]) * u[1],
        -l/J * sin(x[2]) + r/J * u[0]
    ])


def secord_update(t, x, u, params={}):
    """Second order system dynamics"""
    omega0 = params.get('omega0', 1.)
    zeta = params.get('zeta', 0.5)
    return np.array([
        x[1],
        -2 * zeta * omega0 * x[1] - omega0*omega0 * x[0] + u[0]
    ])


def secord_output(t, x, u, params={}):
    """Second order system dynamics output"""
    return np.array([x[0]])


def test_interconnect_name():
    g = ct.LinearIOSystem(ct.ss(-1,1,1,0),
                          inputs=['u'],
                          outputs=['y'],
                          name='g')
    k = ct.LinearIOSystem(ct.ss(0,10,2,0),
                          inputs=['e'],
                          outputs=['z'],
                          name='k')
    h = ct.interconnect([g,k],
                            inputs=['u','e'],
                            outputs=['y','z'])
    assert re.match(r'sys\[\d+\]', h.name), f"Interconnect default name does not match 'sys[]' pattern, got '{h.name}'"

    h = ct.interconnect([g,k],
                            inputs=['u','e'],
                            outputs=['y','z'],
                            name='ic_system')
    assert h.name == 'ic_system', f"Interconnect name excpected 'ic_system', got '{h.name}'"


def test_interconnect_unused_input():
    # test that warnings about unused inputs are reported, or not,
    # as required
    g = ct.LinearIOSystem(ct.ss(-1,1,1,0),
                          inputs=['u'],
                          outputs=['y'],
                          name='g')

    s = ct.summing_junction(inputs=['r','-y','-n'],
                            outputs=['e'],
                            name='s')

    k = ct.LinearIOSystem(ct.ss(0,10,2,0),
                          inputs=['e'],
                          outputs=['u'],
                          name='k')

    with pytest.warns(
            UserWarning, match=r"Unused input\(s\) in InterconnectedSystem"):
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'])

    with warnings.catch_warnings():
        # no warning if output explicitly ignored, various argument forms
        warnings.simplefilter("error")
        # strip out matrix warnings
        warnings.filterwarnings("ignore", "the matrix subclass",
                                category=PendingDeprecationWarning)
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_inputs=['n'])

        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_inputs=['s.n'])

        # no warning if auto-connect disabled
        h = ct.interconnect([g,s,k],
                            connections=False)

    # warn if explicity ignored input in fact used
    with pytest.warns(
            UserWarning,
            match=r"Input\(s\) specified as ignored is \(are\) used:") \
            as record:
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_inputs=['u','n'])

    with pytest.warns(
            UserWarning,
            match=r"Input\(s\) specified as ignored is \(are\) used:") \
            as record:
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_inputs=['k.e','n'])

    # error if ignored signal doesn't exist
    with pytest.raises(ValueError):
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_inputs=['v'])


def test_interconnect_unused_output():
    # test that warnings about ignored outputs are reported, or not,
    # as required
    g = ct.LinearIOSystem(ct.ss(-1,1,[[1],[-1]],[[0],[1]]),
                          inputs=['u'],
                          outputs=['y','dy'],
                          name='g')

    s = ct.summing_junction(inputs=['r','-y'],
                            outputs=['e'],
                            name='s')

    k = ct.LinearIOSystem(ct.ss(0,10,2,0),
                          inputs=['e'],
                          outputs=['u'],
                          name='k')

    with pytest.warns(
            UserWarning,
            match=r"Unused output\(s\) in InterconnectedSystem:") as record:
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'])


    # no warning if output explicitly ignored
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # strip out matrix warnings
        warnings.filterwarnings("ignore", "the matrix subclass",
                                category=PendingDeprecationWarning)
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_outputs=['dy'])

        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_outputs=['g.dy'])

        # no warning if auto-connect disabled
        h = ct.interconnect([g,s,k],
                            connections=False)

    # warn if explicity ignored output in fact used
    with pytest.warns(
            UserWarning,
            match=r"Output\(s\) specified as ignored is \(are\) used:"):
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_outputs=['dy','u'])

    with pytest.warns(
            UserWarning,
            match=r"Output\(s\) specified as ignored is \(are\) used:"):
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_outputs=['dy', ('k.u')])

    # error if ignored signal doesn't exist
    with pytest.raises(ValueError):
        h = ct.interconnect([g,s,k],
                            inputs=['r'],
                            outputs=['y'],
                            ignore_outputs=['v'])


def test_interconnect_add_unused():
    P = ct.ss(
        [[-1]], [[1, -1]], [[-1], [1]], 0,
        inputs=['u1', 'u2'], outputs=['y1','y2'], name='g')
    S = ct.summing_junction(inputs=['r','-y1'], outputs=['e'], name='s')
    C = ct.ss(0, 10, 2, 0, inputs=['e'], outputs=['u1'], name='k')

    # Try a normal interconnection
    G1 = ct.interconnect(
        [P, S, C], inputs=['r', 'u2'], outputs=['y1', 'y2'])

    # Same system, but using add_unused
    G2 = ct.interconnect(
        [P, S, C], inputs=['r'], outputs=['y1'], add_unused=True)
    assert G2.input_labels == G1.input_labels
    assert G2.input_offset == G1.input_offset
    assert G2.output_labels == G1.output_labels
    assert G2.output_offset == G1.output_offset

    # Ignore one of the inputs
    G3 = ct.interconnect(
        [P, S, C], inputs=['r'], outputs=['y1'], add_unused=True,
        ignore_inputs=['u2'])
    assert G3.input_labels == G1.input_labels[0:1]
    assert G3.output_labels == G1.output_labels
    assert G3.output_offset == G1.output_offset

    # Ignore one of the outputs
    G4 = ct.interconnect(
        [P, S, C], inputs=['r'], outputs=['y1'], add_unused=True,
        ignore_outputs=['y2'])
    assert G4.input_labels == G1.input_labels
    assert G4.input_offset == G1.input_offset
    assert G4.output_labels == G1.output_labels[0:1]


def test_input_output_broadcasting():
    # Create a system, time vector, and noisy input
    sys = ct.rss(6, 2, 3)
    T = np.linspace(0, 10, 10)
    U = np.zeros((sys.ninputs, T.size))
    U[0, :] = np.sin(T)
    U[1, :] = np.zeros_like(U[1, :])
    U[2, :] = np.ones_like(U[2, :])
    X0 = np.array([1, 2])
    P0 = np.array([[3.11, 3.12], [3.21, 3.3]])

    # Simulate the system with nominal input to establish baseline
    resp_base = ct.input_output_response(
        sys, T, U, np.hstack([X0, P0.reshape(-1)]))

    # Split up the inputs into two pieces
    resp_inp1 = ct.input_output_response(sys, T, [U[:1], U[1:]], [X0, P0])
    np.testing.assert_equal(resp_base.states, resp_inp1.states)

    # Specify two of the inputs as constants
    resp_inp2 = ct.input_output_response(sys, T, [U[0], 0, 1], [X0, P0])
    np.testing.assert_equal(resp_base.states, resp_inp2.states)

    # Specify two of the inputs as constant vector
    resp_inp3 = ct.input_output_response(sys, T, [U[0], [0, 1]], [X0, P0])
    np.testing.assert_equal(resp_base.states, resp_inp3.states)

    # Specify only some of the initial conditions
    resp_init = ct.input_output_response(sys, T, [U[0], [0, 1]], [X0, 0])
    resp_cov0 = ct.input_output_response(sys, T, U, [X0, P0 * 0])
    np.testing.assert_equal(resp_cov0.states, resp_init.states)

    # Specify only some of the initial conditions
    with pytest.warns(UserWarning, match="initial state too short; padding"):
        resp_short = ct.input_output_response(sys, T, [U[0], [0, 1]], [X0, 1])

    # Make sure that inconsistent settings don't work
    with pytest.raises(ValueError, match="inconsistent"):
        resp_bad = ct.input_output_response(
            sys, T, (U[0, :], U[:2, :-1]), [X0, P0])

@pytest.mark.parametrize("nstates, ninputs, noutputs", [
    [2, 1, 1],
    [4, 2, 3],
    [0, 1, 1],                  # static function
    [0, 3, 2],                  # static function
])
def test_nonuniform_timepts(nstates, noutputs, ninputs):
    """Test non-uniform time points for simulations"""
    if nstates:
        sys = ct.rss(nstates, noutputs, ninputs)
    else:
        sys = ct.ss(
            [], np.zeros((0, ninputs)), np.zeros((noutputs, 0)),
            np.random.rand(noutputs, ninputs))

    # Start with a uniform set of times
    unifpts = [0, 1, 2, 3, 4,  5,  6,  7,  8,  9, 10]
    uniform = np.outer(
        np.ones(ninputs), [1, 2, 3, 2, 1, -1, -3, -5, -7, -3,  1])
    t_unif, y_unif = ct.input_output_response(
        sys, unifpts, uniform, squeeze=False)

    # Create a non-uniform set of inputs
    noufpts = [0, 2, 4,  8, 10]
    nonunif = np.outer(np.ones(ninputs), [1, 3, 1, -7,  1])
    t_nouf, y_nouf = ct.input_output_response(
        sys, noufpts, nonunif, squeeze=False)

    # Make sure the outputs agree at common times
    np.testing.assert_almost_equal(y_unif[:, noufpts], y_nouf, decimal=6)

    # Resimulate using a new set of evaluation points
    t_even, y_even = ct.input_output_response(
        sys, noufpts, nonunif, t_eval=unifpts, squeeze=False)
    np.testing.assert_almost_equal(y_unif, y_even, decimal=6)


def test_ss_nonlinear():
    """Test ss() for creating nonlinear systems"""
    secord = ct.ss(secord_update, secord_output, inputs='u', outputs='y',
                   states = ['x1', 'x2'], name='secord')
    assert secord.name == 'secord'
    assert secord.input_labels == ['u']
    assert secord.output_labels == ['y']
    assert secord.state_labels == ['x1', 'x2']

    # Make sure we get the same answer for simulations
    T = np.linspace(0, 10, 100)
    U = np.sin(T)
    X0 = np.array([1, -1])
    secord_nlio = ct.NonlinearIOSystem(
        secord_update, secord_output, inputs=1, outputs=1, states=2)
    ss_response = ct.input_output_response(secord, T, U, X0)
    io_response = ct.input_output_response(secord_nlio, T, U, X0)
    np.testing.assert_almost_equal(ss_response.time, io_response.time)
    np.testing.assert_almost_equal(ss_response.inputs, io_response.inputs)
    np.testing.assert_almost_equal(ss_response.outputs, io_response.outputs)

    # Make sure that optional keywords are allowed
    secord = ct.ss(secord_update, secord_output, dt=True)
    assert ct.isdtime(secord)

    # Make sure that state space keywords are flagged
    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.ss(secord_update, remove_useless_states=True)


def test_rss():
    # Basic call, with no arguments
    sys = ct.rss()
    assert sys.ninputs == 1
    assert sys.noutputs == 1
    assert sys.nstates == 1
    assert sys.dt == 0
    assert np.all(np.real(sys.poles()) < 0)

    # Set the timebase explicitly
    sys = ct.rss(inputs=2, outputs=3, states=4, dt=None, name='sys')
    assert sys.name == 'sys'
    assert sys.ninputs == 2
    assert sys.noutputs == 3
    assert sys.nstates == 4
    assert sys.dt == None
    assert np.all(np.real(sys.poles()) < 0)

    # Discrete time
    sys = ct.rss(inputs=['a', 'b'], outputs=1, states=1, dt=True)
    assert sys.ninputs == 2
    assert sys.input_labels == ['a', 'b']
    assert sys.noutputs == 1
    assert sys.nstates == 1
    assert sys.dt == True
    assert np.all(np.abs(sys.poles()) < 1)

    # Call drss directly
    sys = ct.drss(inputs=['a', 'b'], outputs=1, states=1, dt=True)
    assert sys.ninputs == 2
    assert sys.input_labels == ['a', 'b']
    assert sys.noutputs == 1
    assert sys.nstates == 1
    assert sys.dt == True
    assert np.all(np.abs(sys.poles()) < 1)

    with pytest.raises(ValueError, match="continuous timebase"):
        sys = ct.drss(2, 1, 1, dt=0)

    with pytest.warns(UserWarning, match="may be interpreted as continuous"):
        sys = ct.drss(2, 1, 1, dt=None)
        assert np.all(np.abs(sys.poles()) < 1)


def eqpt_rhs(t, x, u, params):
    return np.array([x[0]/2 + u[0], x[0] - x[1]**2 + u[1], x[1] - x[2]])

def eqpt_out(t, x, u, params):
    return np.array([x[0], x[1] + u[1]])

@pytest.mark.parametrize(
    "x0, ix, u0, iu, y0, iy, dx0, idx, dt, x_expect, u_expect", [
        # Equilibrium points with input given
        (0, None, 0, None, None, None, None, None, 0, [0, 0, 0], [0, 0]),
        (0, None, 0, None, None, None, None, None, None, [0, 0, 0], [0, 0]),
        ([0.9, 0.9, 0.9], None, [-1, 0], None, None, None, None, None, 0,
         [2, sqrt(2), sqrt(2)], [-1, 0]),
        ([0.9, -0.9, 0.9], None, [-1, 0], None, None, None, None, None, 0,
         [2, -sqrt(2), -sqrt(2)], [-1, 0]),     # same input, different eqpt
        (0, None, 0, None, None, None, None, None, 1, [0, 0, 0], [0, 0]), #DT
        (0, None, [-1, 0], None, None, None, None, None, 1, None, None),  #DT
        ([0, -0.1, 0], None, [0, -0.25], None, None, None, None, None, 1, #DT
         [0, -0.5, -0.25], [0, -0.25]),

        # Equilibrium points with output given
        ([0.9, 0.9, 0.9], None, [-0.9, 0], None, [2, sqrt(2)], None, None,
         None, 0, [2, sqrt(2), sqrt(2)], [-1, 0]),
        (0, None, [0, -0.25], None, [0, -0.75], None, None, None, 1,      #DT
         [0, -0.5, -0.25], [0, -0.25]),

        # Equilibrium points with mixture of inputs and outputs given
        ([0.9, 0.9, 0.9], None, [-1, 0], [0], [2, sqrt(2)], [1], None,
         None, 0, [2, sqrt(2), sqrt(2)], [-1, 0]),
        (0, None, [0, -0.22], [0], [0, -0.75], [1], None, None, 1,        #DT
         [0, -0.5, -0.25], [0, -0.25]),
    ])

def test_find_eqpt(x0, ix, u0, iu, y0, iy, dx0, idx, dt, x_expect, u_expect):
    sys = ct.NonlinearIOSystem(
        eqpt_rhs, eqpt_out, dt=dt, states=3, inputs=2, outputs=2)

    xeq, ueq = ct.find_eqpt(
        sys, x0, u0, y0, ix=ix, iu=iu, iy=iy, dx0=dx0, idx=idx)

    # If no equilibrium points, skip remaining tests
    if x_expect is None:
        assert xeq is None
        assert ueq is None
        return

    # Make sure we are at an appropriate equilibrium point
    if dt is None or dt == 0:
        # Continuous time system
        np.testing.assert_allclose(eqpt_rhs(0, xeq, ueq, {}), 0, atol=1e-6)
        if y0 is not None:
            y0 = np.array(y0)
            iy = np.s_[:] if iy is None else np.array(iy)
            np.testing.assert_allclose(
                eqpt_out(0, xeq, ueq, {})[iy], y0[iy], atol=1e-6)

    else:
        # Discrete time system
        np.testing.assert_allclose(eqpt_rhs(0, xeq, ueq, {}), xeq, atol=1e-6)
        if y0 is not None:
            y0 = np.array(y0)
            iy = np.s_[:] if iy is None else np.array(iy)
            np.testing.assert_allclose(
                eqpt_out(0, xeq, ueq, {})[iy], y0[iy], atol=1e-6)

    # Check that we got the expected result as well
    np.testing.assert_allclose(np.array(xeq), x_expect, atol=1e-6)
    np.testing.assert_allclose(np.array(ueq), u_expect, atol=1e-6)

def test_iosys_sample():
    csys = ct.rss(2, 1, 1)
    dsys = csys.sample(0.1)
    assert isinstance(dsys, ct.LinearIOSystem)
    assert dsys.dt == 0.1

    csys = ct.rss(2, 1, 1)
    dsys = ct.sample_system(csys, 0.1)
    assert isinstance(dsys, ct.LinearIOSystem)
    assert dsys.dt == 0.1
