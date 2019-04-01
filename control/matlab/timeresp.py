"""
Time response routines in the Matlab compatibility package

Note that the return arguments are different than in the standard control package.
"""

__all__ = ['step', 'stepinfo', 'impulse', 'initial', 'lsim']

def step(sys, T=None, X0=0., input=0, output=None, return_x=False):
    '''
    Step response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation.  Optionally, one output may be
    selected. If no selection is made for the output, all outputs are
    given. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        Index of the input that will be used in this simulation.

    output: int
        If given, index of the output that is returned by this simulation.

    Returns
    -------
    yout: array
        Response of the system

    T: array
        Time values of the output

    xout: array (if selected)
        Individual response of each x variable



    See Also
    --------
    lsim, initial, impulse

    Examples
    --------
    >>> yout, T = step(sys, T, X0)
    '''
    from ..timeresp import step_response

    T, yout, xout = step_response(sys, T, X0, input, output,
                                  transpose = True, return_x=True)

    if return_x:
        return yout, T, xout

    return yout, T

def stepinfo(sys, T=None, SettlingTimeThreshold=0.02, RiseTimeLimits=(0.1,0.9)):
    '''
    Step response characteristics (Rise time, Settling Time, Peak and others).

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    SettlingTimeThreshold: float value, optional
        Defines the error to compute settling time (default = 0.02)

    RiseTimeLimits: tuple (lower_threshold, upper_theshold)
        Defines the lower and upper threshold for RiseTime computation

    Returns
    -------
    S: a dictionary containing:
        RiseTime: Time from 10% to 90% of the steady-state value.
        SettlingTime: Time to enter inside a default error of 2%
        SettlingMin: Minimum value after RiseTime
        SettlingMax: Maximum value after RiseTime
        Overshoot: Percentage of the Peak relative to steady value
        Undershoot: Percentage of undershoot
        Peak: Absolute peak value
        PeakTime: time of the Peak
        SteadyStateValue: Steady-state value


    See Also
    --------
    step, lsim, initial, impulse

    Examples
    --------
    >>> S = stepinfo(sys, T)
    '''
    from ..timeresp import step_info

    S = step_info(sys, T, SettlingTimeThreshold, RiseTimeLimits)

    return S

def impulse(sys, T=None, X0=0., input=0, output=None, return_x=False):
    '''
    Impulse response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation.  Optionally, one output may be
    selected. If no selection is made for the output, all outputs are
    given. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    Parameters
    ----------
    sys: StateSpace, TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        Index of the input that will be used in this simulation.

    output: int
        Index of the output that will be used in this simulation.

    Returns
    -------
    yout: array
        Response of the system

    T: array
        Time values of the output

    xout: array (if selected)
        Individual response of each x variable

    See Also
    --------
    lsim, step, initial

    Examples
    --------
    >>> yout, T = impulse(sys, T)
    '''
    from ..timeresp import impulse_response
    T, yout, xout = impulse_response(sys, T, X0, input, output,
                                     transpose = True, return_x=True)

    if return_x:
        return yout, T, xout

    return yout, T

def initial(sys, T=None, X0=0., input=None, output=None, return_x=False):
    '''
    Initial condition response of a linear system

    If the system has multiple outputs (?IMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate

    T: array-like object, optional
        Time vector (argument is autocomputed if not given)

    X0: array-like object or number, optional
        Initial condition (default = 0)

        Numbers are converted to constant arrays with the correct shape.

    input: int
        This input is ignored, but present for compatibility with step
        and impulse.

    output: int
        If given, index of the output that is returned by this simulation.

    Returns
    -------
    yout: array
        Response of the system

    T: array
        Time values of the output

    xout: array (if selected)
        Individual response of each x variable

    See Also
    --------
    lsim, step, impulse

    Examples
    --------
    >>> yout, T = initial(sys, T, X0)

    '''
    from ..timeresp import initial_response
    T, yout, xout = initial_response(sys, T, X0, output=output,
                                     transpose=True, return_x=True)

    if return_x:
        return yout, T, xout

    return yout, T

def lsim(sys, U=0., T=None, X0=0.):
    '''
    Simulate the output of a linear system.

    As a convenience for parameters `U`, `X0`:
    Numbers (scalars) are converted to constant arrays with the correct shape.
    The correct shape is inferred from arguments `sys` and `T`.

    Parameters
    ----------
    sys: LTI (StateSpace, or TransferFunction)
        LTI system to simulate

    U: array-like or number, optional
        Input array giving input at each time `T` (default = 0).

        If `U` is ``None`` or ``0``, a special algorithm is used. This special
        algorithm is faster than the general algorithm, which is used otherwise.

    T: array-like
        Time steps at which the input is defined, numbers must be (strictly
        monotonic) increasing.

    X0: array-like or number, optional
        Initial condition (default = 0).

    Returns
    -------
    yout: array
        Response of the system.
    T: array
        Time values of the output.
    xout: array
        Time evolution of the state vector.

    See Also
    --------
    step, initial, impulse

    Examples
    --------
    >>> yout, T, xout = lsim(sys, U, T, X0)
    '''
    from ..timeresp import forced_response
    T, yout, xout = forced_response(sys, T, U, X0, transpose = True)
    return yout, T, xout
