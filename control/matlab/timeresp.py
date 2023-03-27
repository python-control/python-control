"""
Time response routines in the Matlab compatibility package

Note that the return arguments are different than in the standard control package.
"""

__all__ = ['step', 'stepinfo', 'impulse', 'initial', 'lsim']

def step(sys, T=None, X0=0., input=0, output=None, return_x=False):
    '''Step response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation.  Optionally, one output may be
    selected. If no selection is made for the output, all outputs are
    given. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate
    T: array-like or number, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given)
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
    >>> from control.matlab import step, rss

    >>> G = rss(4)
    >>> yout, T = step(G)

    '''
    from ..timeresp import step_response

    # Switch output argument order and transpose outputs
    out = step_response(sys, T, X0, input, output,
                        transpose=True, return_x=return_x)
    return (out[1], out[0], out[2]) if return_x else (out[1], out[0])


def stepinfo(sysdata, T=None, yfinal=None, SettlingTimeThreshold=0.02,
             RiseTimeLimits=(0.1, 0.9)):
    """Step response characteristics (Rise time, Settling Time, Peak and others)

    Parameters
    ----------
    sysdata : StateSpace or TransferFunction or array_like
        The system data. Either LTI system to similate (StateSpace,
        TransferFunction), or a time series of step response data.
    T : array_like or float, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given).
        Required, if sysdata is a time series of response data.
    yfinal : scalar or array_like, optional
        Steady-state response. If not given, sysdata.dcgain() is used for
        systems to simulate and the last value of the the response data is
        used for a given time series of response data. Scalar for SISO,
        (noutputs, ninputs) array_like for MIMO systems.
    SettlingTimeThreshold : float, optional
        Defines the error to compute settling time (default = 0.02)
    RiseTimeLimits : tuple (lower_threshold, upper_theshold)
        Defines the lower and upper threshold for RiseTime computation

    Returns
    -------
    S : dict or list of list of dict
        If `sysdata` corresponds to a SISO system, S is a dictionary
        containing:

        RiseTime:
            Time from 10% to 90% of the steady-state value.
        SettlingTime:
            Time to enter inside a default error of 2%
        SettlingMin:
            Minimum value after RiseTime
        SettlingMax:
            Maximum value after RiseTime
        Overshoot:
            Percentage of the Peak relative to steady value
        Undershoot:
            Percentage of undershoot
        Peak:
            Absolute peak value
        PeakTime:
            time of the Peak
        SteadyStateValue:
            Steady-state value

        If `sysdata` corresponds to a MIMO system, `S` is a 2D list of dicts.
        To get the step response characteristics from the j-th input to the
        i-th output, access ``S[i][j]``


    See Also
    --------
    step, lsim, initial, impulse

    Examples
    --------
    >>> from control.matlab import stepinfo, rss

    >>> G = rss(4)
    >>> S = stepinfo(G)

    """
    from ..timeresp import step_info

    # Call step_info with MATLAB defaults
    S = step_info(sysdata, T=T, T_num=None, yfinal=yfinal,
                  SettlingTimeThreshold=SettlingTimeThreshold,
                  RiseTimeLimits=RiseTimeLimits)

    return S

def impulse(sys, T=None, X0=0., input=0, output=None, return_x=False):
    '''Impulse response of a linear system

    If the system has multiple inputs or outputs (MIMO), one input has
    to be selected for the simulation.  Optionally, one output may be
    selected. If no selection is made for the output, all outputs are
    given. The parameters `input` and `output` do this. All other
    inputs are set to 0, all other outputs are ignored.

    Parameters
    ----------
    sys: StateSpace, TransferFunction
        LTI system to simulate
    T: array-like or number, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given)
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
    >>> from control.matlab import rss, impulse

    >>> G = rss()
    >>> yout, T = impulse(G)

    '''
    from ..timeresp import impulse_response

    # Switch output argument order and transpose outputs
    out = impulse_response(sys, T, X0, input, output,
                           transpose = True, return_x=return_x)
    return (out[1], out[0], out[2]) if return_x else (out[1], out[0])

def initial(sys, T=None, X0=0., input=None, output=None, return_x=False):
    '''Initial condition response of a linear system

    If the system has multiple outputs (?IMO), optionally, one output
    may be selected. If no selection is made for the output, all
    outputs are given.

    Parameters
    ----------
    sys: StateSpace, or TransferFunction
        LTI system to simulate
    T: array-like or number, optional
        Time vector, or simulation time duration if a number (time vector is
        autocomputed if not given)
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
    >>> from control.matlab import initial, rss

    >>> G = rss(4)
    >>> yout, T = initial(G)

    '''
    from ..timeresp import initial_response

    # Switch output argument order and transpose outputs
    T, yout, xout = initial_response(sys, T, X0, output=output,
                                     transpose=True, return_x=True)
    return (yout, T, xout) if return_x else (yout, T)


def lsim(sys, U=0., T=None, X0=0.):
    '''Simulate the output of a linear system

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
    T: array-like, optional for discrete LTI `sys`
        Time steps at which the input is defined; values must be evenly spaced.
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
    >>> from control.matlab import rss, lsim

    >>> G = rss(4)
    >>> T = np.linspace(0,10)
    >>> yout, T, xout = lsim(G, T=T)

    '''
    from ..timeresp import forced_response

    # Switch output argument order and transpose outputs (and always return x)
    out = forced_response(sys, T, U, X0, return_x=True, transpose=True)
    return out[1], out[0], out[2]
