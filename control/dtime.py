# dtime.py - functions for manipulating discrete-time systems
#
# Initial author: Richard M. Murray
# Creation date: 6 October 2012

"""Functions for manipulating discrete-time systems."""

from .iosys import isctime

__all__ = ['sample_system', 'c2d']

# Sample a continuous-time system
def sample_system(sysc, Ts, method='zoh', alpha=None, prewarp_frequency=None,
        name=None, copy_names=True, **kwargs):
    """Convert a continuous-time system to discrete time by sampling.

    Parameters
    ----------
    sysc : `StateSpace` or `TransferFunction`
        Continuous time system to be converted.
    Ts : float > 0
        Sampling period.
    method : string
        Method to use for conversion, e.g. 'bilinear', 'zoh' (default).
    alpha : float within [0, 1]
        The generalized bilinear transformation weighting parameter, which
        should only be specified with method="gbt", and is ignored
        otherwise. See `scipy.signal.cont2discrete`.
    prewarp_frequency : float within [0, infinity)
        The frequency [rad/s] at which to match with the input continuous-
        time system's magnitude and phase (only valid for method='bilinear',
        'tustin', or 'gbt' with alpha=0.5).

    Returns
    -------
    sysd : LTI of the same class (`StateSpace` or `TransferFunction`)
        Discrete time system, with sampling rate `Ts`.

    Other Parameters
    ----------------
    inputs : int, list of str or None, optional
        Description of the system inputs.  If not specified, the original
        system inputs are used.  See `InputOutputSystem` for more
        information.
    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None, optional
        Description of the system states.  Same format as `inputs`. Only
        available if the system is `StateSpace`.
    name : string, optional
        Set the name of the sampled system.  If not specified and
        if `copy_names` is False, a generic name 'sys[id]' is generated
        with a unique integer id.  If `copy_names` is True, the new system
        name is determined by adding the prefix and suffix strings in
        `config.defaults['iosys.sampled_system_name_prefix']` and
        `config.defaults['iosys.sampled_system_name_suffix']`, with the
        default being to add the suffix '$sampled'.
    copy_names : bool, optional
        If True, copy the names of the input signals, output
        signals, and states to the sampled system.

    Notes
    -----
    See `StateSpace.sample` or `TransferFunction.sample` for further
    details on implementation for state space and transfer function
    systems, including available methods.

    Examples
    --------
    >>> Gc = ct.tf([1], [1, 2, 1])
    >>> Gc.isdtime()
    False
    >>> Gd = ct.sample_system(Gc, 1, method='bilinear')
    >>> Gd.isdtime()
    True

    """

    # Make sure we have a continuous-time system
    if not isctime(sysc):
        raise ValueError("First argument must be continuous-time system")

    return sysc.sample(Ts,
        method=method, alpha=alpha, prewarp_frequency=prewarp_frequency,
        name=name, copy_names=copy_names, **kwargs)


# Convenience aliases
c2d = sample_system
