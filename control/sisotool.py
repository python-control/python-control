# sisotool.py - interactive tool for SISO control design

"""Interactive tool for SISO control design."""

__all__ = ['sisotool', 'rootlocus_pid_designer']

import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from control.exception import ControlMIMONotImplemented
from control.statesp import _convert_to_statespace

from . import config
from .bdalg import append, connect
from .freqplot import bode_plot
from .iosys import common_timebase, isctime, isdtime
from .lti import frequency_response
from .nlsys import interconnect
from .statesp import ss, summing_junction
from .timeresp import step_response
from .xferfcn import tf
from .rlocus import add_loci_recalculate

_sisotool_defaults = {
    'sisotool.initial_gain': 1
}

def sisotool(sys, initial_gain=None, xlim_rlocus=None, ylim_rlocus=None,
             plotstr_rlocus='C0', rlocus_grid=False, omega=None, dB=None,
             Hz=None, deg=None, omega_limits=None, omega_num=None,
             margins_bode=True, tvect=None, kvect=None):
    """Collection of plots inspired by MATLAB's sisotool.

    The left two plots contain the bode magnitude and phase diagrams.
    The top right plot is a clickable root locus plot, clicking on the
    root locus will change the gain of the system. The bottom left plot
    shows a closed loop time response.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems. If `sys` is SISO, use the same system
        for the root locus and step response. If it is desired to see a
        different step response than ``feedback(K*sys, 1)``, such as a
        disturbance response, `sys` can be provided as a two-input,
        two-output system. For two-input, two-output system, sisotool
        inserts the negative of the selected gain `K` between the first
        output and first input and uses the second input and output for
        computing the step response. To see the disturbance response,
        configure your plant to have as its second input the disturbance
        input.  To view the step response with a feedforward controller,
        give your plant two identical inputs, and sum your feedback
        controller and your feedforward controller and multiply them into
        your plant's second input. It is also possible to accommodate a
        system with a gain in the feedback.
    initial_gain : float, optional
        Initial gain to use for plotting root locus. Defaults to 1
        (`config.defaults['sisotool.initial_gain']`).
    xlim_rlocus : tuple or list, optional
        Control of x-axis range (see `matplotlib.axes.Axes.set_xlim`).
    ylim_rlocus : tuple or list, optional
        Control of y-axis range (see `matplotlib.axes.Axes.set_ylim`).
    plotstr_rlocus : `matplotlib.pyplot.plot` format string, optional
        Plotting style for the root locus plot(color, linestyle, etc).
    rlocus_grid : boolean (default = False)
        If True, plot s- or z-plane grid.
    omega : array_like
        List of frequencies in rad/sec to be used for bode plot.
    dB : boolean
        If True, plot result in dB for the bode plot.
    Hz : boolean
        If True, plot frequency in Hz for the bode plot (omega must be
        provided in rad/sec).
    deg : boolean
        If True, plot phase in degrees for the bode plot (else radians).
    omega_limits : array_like of two values
        Limits of the to generate frequency vector.  If Hz=True the limits
        are in Hz otherwise in rad/s. Ignored if omega is provided, and
        auto-generated if omitted.
    omega_num : int
        Number of samples to plot.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.
    margins_bode : boolean
        If True, plot gain and phase margin in the bode plot.
    tvect : list or ndarray, optional
        List of time steps to use for closed loop step response.

    Examples
    --------
    >>> G = ct.tf([1000], [1, 25, 100, 0])
    >>> ct.sisotool(G)                                          # doctest: +SKIP

    """
    from .rlocus import root_locus_map

    # sys as loop transfer function if SISO
    if not sys.issiso():
        if not (sys.ninputs == 2 and sys.noutputs == 2):
            raise ControlMIMONotImplemented(
                'sys must be SISO or 2-input, 2-output')

    # Setup sisotool figure or superimpose if one is already present
    fig = plt.gcf()
    if fig.canvas.manager.get_window_title() != 'Sisotool':
        plt.close(fig)
        fig, axes = plt.subplots(2, 2)
        fig.canvas.manager.set_window_title('Sisotool')
    else:
        axes = np.array(fig.get_axes()).reshape(2, 2)

    # Extract bode plot parameters
    bode_plot_params = {
        'omega': omega,
        'dB': dB,
        'Hz': Hz,
        'deg': deg,
        'omega_limits': omega_limits,
        'omega_num' : omega_num,
        'ax': axes[:, 0:1],
        'display_margins': 'overlay' if margins_bode else False,
    }

    # Check to see if legacy 'PrintGain' keyword was used
    if kvect is not None:
        warnings.warn("'kvect' keyword is deprecated in sisotool; "
                      "use 'initial_gain' instead", FutureWarning)
        initial_gain = np.atleast_1d(kvect)[0]
    initial_gain = config._get_param('sisotool', 'initial_gain',
            initial_gain, _sisotool_defaults)

    # First time call to setup the Bode and step response plots
    _SisotoolUpdate(sys, fig, initial_gain, bode_plot_params)

    # root_locus(
    #     sys[0, 0], initial_gain=initial_gain, xlim=xlim_rlocus,
    #     ylim=ylim_rlocus, plotstr=plotstr_rlocus, grid=rlocus_grid,
    #     ax=fig.axes[1])
    ax_rlocus = axes[0,1]  # fig.axes[1]
    cplt = root_locus_map(sys[0, 0]).plot(
        xlim=xlim_rlocus, ylim=ylim_rlocus,
        initial_gain=initial_gain, ax=ax_rlocus)
    if rlocus_grid is False:
        # Need to generate grid manually, since root_locus_plot() won't
        from .grid import nogrid
        nogrid(sys.dt, ax=ax_rlocus)

    # install a zoom callback on the root-locus axis
    add_loci_recalculate(sys, cplt, ax_rlocus)

    # Reset the button release callback so that we can update all plots
    fig.canvas.mpl_connect(
        'button_release_event', partial(
            _click_dispatcher, sys=sys, ax=fig.axes[1],
            bode_plot_params=bode_plot_params, tvect=tvect))


def _click_dispatcher(event, sys, ax, bode_plot_params, tvect):
    # Zoom handled by specialized callback in rlocus, only handle gain plot
    if event.inaxes == ax.axes:

        fig = ax.figure

        # if a point is clicked on the rootlocus plot visually emphasize it
        # K = _RLFeedbackClicksPoint(
        #     event, sys, fig, ax_rlocus, show_clicked=True)
        from .pzmap import _create_root_locus_label, _find_root_locus_gain, \
            _mark_root_locus_gain

        K, s = _find_root_locus_gain(event, sys, ax)
        if K is not None:
            _mark_root_locus_gain(ax, sys, K)
            fig.suptitle(_create_root_locus_label(sys, K, s), fontsize=10)
            _SisotoolUpdate(sys, fig, K, bode_plot_params, tvect)

        # Update the canvas
        fig.canvas.draw()


def _SisotoolUpdate(sys, fig, K, bode_plot_params, tvect=None):

    title_font_size = 10
    label_font_size = 8

    # Get the subaxes and clear them
    ax_mag, ax_rlocus, ax_phase, ax_step = \
        fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]

    # Catch matplotlib 2.1.x and higher userwarnings when clearing a log axis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax_step.clear(), ax_mag.clear(), ax_phase.clear()

    sys_loop = sys if sys.issiso() else sys[0,0]

    # Update the Bode plot
    bode_plot_params['data'] = frequency_response(sys_loop*K.real)
    bode_plot(**bode_plot_params, title=False)

    # Set the titles and labels
    ax_mag.set_title('Bode magnitude',fontsize = title_font_size)
    ax_mag.set_ylabel(ax_mag.get_ylabel(), fontsize=label_font_size)
    ax_mag.tick_params(axis='both', which='major', labelsize=label_font_size)

    ax_phase.set_title('Bode phase',fontsize=title_font_size)
    ax_phase.set_xlabel(ax_phase.get_xlabel(),fontsize=label_font_size)
    ax_phase.set_ylabel(ax_phase.get_ylabel(),fontsize=label_font_size)
    ax_phase.get_xaxis().set_label_coords(0.5, -0.15)
    ax_phase.tick_params(axis='both', which='major', labelsize=label_font_size)

    if not ax_phase.get_shared_x_axes().joined(ax_phase, ax_mag):
        ax_phase.sharex(ax_mag)

    ax_step.set_title('Step response',fontsize = title_font_size)
    ax_step.set_xlabel('Time (seconds)',fontsize=label_font_size)
    ax_step.set_ylabel('Output',fontsize=label_font_size)
    ax_step.get_xaxis().set_label_coords(0.5, -0.15)
    ax_step.get_yaxis().set_label_coords(-0.15, 0.5)
    ax_step.tick_params(axis='both', which='major', labelsize=label_font_size)

    ax_rlocus.set_title('Root locus',fontsize = title_font_size)
    ax_rlocus.set_ylabel('Imag', fontsize=label_font_size)
    ax_rlocus.set_xlabel('Real', fontsize=label_font_size)
    ax_rlocus.get_xaxis().set_label_coords(0.5, -0.15)
    ax_rlocus.get_yaxis().set_label_coords(-0.15, 0.5)
    ax_rlocus.tick_params(axis='both', which='major',labelsize=label_font_size)

    # Generate the step response and plot it
    if sys.issiso():
        sys_closed = (K*sys).feedback(1)
    else:
        sys_closed = append(sys, -K)
        connects = [[1, 3],
                    [3, 1]]
        # Filter out known warning due to use of connect
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message="`connect`", category=DeprecationWarning)
            sys_closed = connect(sys_closed, connects, 2, 2)
    if tvect is None:
        tvect, yout = step_response(sys_closed, T_num=100)
    else:
        tvect, yout = step_response(sys_closed, tvect)
    if isdtime(sys_closed, strict=True):
        ax_step.plot(tvect, yout, '.')
    else:
        ax_step.plot(tvect, yout)
    ax_step.axhline(1.,linestyle=':',color='k',zorder=-20)

    # Manually adjust the spacing and draw the canvas
    fig.subplots_adjust(top=0.9,wspace = 0.3,hspace=0.35)
    fig.canvas.draw()

# contributed by Sawyer Fuller, minster@uw.edu 2021.11.02, based on
# an implementation in Matlab by Martin Berg.
def rootlocus_pid_designer(plant, gain='P', sign=+1, input_signal='r',
                           Kp0=0, Ki0=0, Kd0=0, deltaK=0.001, tau=0.01,
                           C_ff=0, derivative_in_feedback_path=False,
                           plot=True):
    """Manual PID controller design based on root locus using Sisotool.

    Uses `sisotool` to investigate the effect of adding or subtracting an
    amount `deltaK` to the proportional, integral, or derivative (PID) gains
    of a controller. One of the PID gains, `Kp`, `Ki`, or `Kd`, respectively,
    can be modified at a time. `sisotool` plots the step response, frequency
    response, and root locus of the closed-loop system controlling the
    dynamical system specified by `plant`. Can be used with either non-
    interactive plots (e.g. in a Jupyter Notebook), or interactive plots.

    To use non-interactively, choose starting-point PID gains `Kp0`, `Ki0`,
    and `Kd0` (you might want to start with all zeros to begin with),
    select which gain you would like to vary (e.g. `gain` = 'P', 'I',
    or 'D'), and choose a value of `deltaK` (default 0.001) to specify
    by how much you would like to change that gain. Repeatedly run
    `rootlocus_pid_designer` with different values of `deltaK` until you
    are satisfied with the performance for that gain. Then, to tune a
    different gain, e.g. 'I', make sure to add your chosen `deltaK` to
    the previous gain you you were tuning.

    Example: to examine the effect of varying `Kp` starting from an initial
    value of 10, use the arguments ``gain='P', Kp0=10`` and try varying values
    of `deltaK`. Suppose a `deltaK` of 5 gives satisfactory performance. Then,
    to tune the derivative gain, add your selected `deltaK` to `Kp0` in the
    next call using the arguments ``gain='D', Kp0=15``, to see how adding
    different values of `deltaK` to your derivative gain affects performance.

    To use with interactive plots, you will need to enable interactive mode
    if you are in a Jupyter Notebook, e.g. using ``%matplotlib``. See
    `Interactive Plots <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ion.html>`_
    for more information. Click on a branch of the root locus plot to try
    different values of `deltaK`. Each click updates plots and prints the
    corresponding `deltaK`. It may be helpful to zoom in using the magnifying
    glass on the plot to get more locations to click. Just make sure to
    deactivate magnification mode when you are done by clicking the magnifying
    glass. Otherwise you will not be able to be able to choose a gain on the
    root locus plot. When you are done, ``%matplotlib inline`` returns to
    inline, non-interactive plotting.

    By default, all three PID terms are in the forward path C_f in the
    diagram shown below, that is,

    C_f = Kp + Ki/s + Kd*s/(tau*s + 1).

    ::

          ------> C_ff ------    d
          |                 |    |
      r   |     e           V    V  u         y
      ------->O---> C_f --->O--->O---> plant --->
              ^-            ^-                |
              |             |                 |
              |             ----- C_b <-------|
              ---------------------------------

    If `plant` is a discrete-time system, then the proportional, integral,
    and derivative terms are given instead by Kp, Ki*dt/2*(z+1)/(z-1), and
    Kd/dt*(z-1)/z, respectively.

    It is also possible to move the derivative term into the feedback path
    `C_b` using `derivative_in_feedback_path` = True. This may be desired to
    avoid that the plant is subject to an impulse function when the reference
    `r` is a step input. `C_b` is otherwise set to zero.

    If `plant` is a 2-input system, the disturbance `d` is fed directly into
    its second input rather than being added to `u`.

    Parameters
    ----------
    plant : `LTI` (`TransferFunction` or `StateSpace` system)
        The dynamical system to be controlled.
    gain : string, optional
        Which gain to vary by `deltaK`. Must be one of 'P', 'I', or 'D'
        (proportional, integral, or derivative).
    sign : int, optional
        The sign of deltaK gain perturbation.
    input_signal : string, optional
        The input used for the step response; must be 'r' (reference) or
        'd' (disturbance) (see figure above).
    Kp0, Ki0, Kd0 : float, optional
        Initial values for proportional, integral, and derivative gains,
        respectively.
    deltaK : float, optional
        Perturbation value for gain specified by the `gain` keyword.
    tau : float, optional
        The time constant associated with the pole in the continuous-time
        derivative term. This is required to make the derivative transfer
        function proper.
    C_ff : float or `LTI` system, optional
        Feedforward controller. If `LTI`, must have timebase that is
        compatible with plant.
    derivative_in_feedback_path : bool, optional
        Whether to place the derivative term in feedback transfer function
        `C_b` instead of the forward transfer function `C_f`.
    plot : bool, optional
        Whether to create Sisotool interactive plot.

    Returns
    -------
    closedloop : `StateSpace` system
        The closed-loop system using initial gains.

    Notes
    -----
    When running using iPython or Jupyter, use ``%matplotlib`` to configure
    the session for interactive support.

    """

    if plant.ninputs == 1:
        plant = ss(plant, inputs='u', outputs='y')
    elif plant.ninputs == 2:
        plant = ss(plant, inputs=['u', 'd'], outputs='y')
    else:
        raise ValueError("plant must have one or two inputs")
    C_ff = ss(_convert_to_statespace(C_ff),   inputs='r', outputs='uff')
    dt = common_timebase(plant, C_ff)

    # create systems used for interconnections
    e_summer = summing_junction(['r', '-y'], 'e')
    if plant.ninputs == 2:
        u_summer = summing_junction(['ufb', 'uff'], 'u')
    else:
        u_summer = summing_junction(['ufb', 'uff', 'd'], 'u')

    if isctime(plant):
        prop  = tf(1, 1, inputs='e', outputs='prop_e')
        integ = tf(1, [1, 0], inputs='e', outputs='int_e')
        deriv = tf([1, 0], [tau, 1], inputs='y', outputs='deriv')
    else: # discrete time
        prop  = tf(1, 1, dt, inputs='e', outputs='prop_e')
        integ = tf([dt/2, dt/2], [1, -1], dt, inputs='e', outputs='int_e')
        deriv = tf([1, -1], [dt, 0], dt, inputs='y', outputs='deriv')

    if derivative_in_feedback_path:
        deriv = -deriv
        deriv.input_labels = 'e'

    # create gain blocks
    Kpgain = tf(Kp0, 1, inputs='prop_e', outputs='ufb')
    Kigain = tf(Ki0, 1, inputs='int_e', outputs='ufb')
    Kdgain = tf(Kd0, 1, inputs='deriv', outputs='ufb')

    # for the gain that is varied, replace gain block with a special block
    # that has an 'input' and an 'output' that creates loop transfer function
    if gain in ('P', 'p'):
        Kpgain = ss([],[],[],[[0, 1], [-sign, Kp0]],
            inputs=['input', 'prop_e'], outputs=['output', 'ufb'])
    elif gain in ('I', 'i'):
        Kigain = ss([],[],[],[[0, 1], [-sign, Ki0]],
            inputs=['input', 'int_e'],  outputs=['output', 'ufb'])
    elif gain in ('D', 'd'):
        Kdgain = ss([],[],[],[[0, 1], [-sign, Kd0]],
            inputs=['input', 'deriv'], outputs=['output', 'ufb'])
    else:
        raise ValueError(gain + ' gain not recognized.')

    # the second input and output are used by sisotool to plot step response
    loop = interconnect((plant, Kpgain, Kigain, Kdgain, prop, integ, deriv,
                            C_ff, e_summer, u_summer),
                            inplist=['input', input_signal],
                            outlist=['output', 'y'], check_unused=False)
    if plot:
        sisotool(loop, initial_gain=deltaK)
    cl = loop[1, 1] # closed loop transfer function with initial gains
    return ss(cl.A, cl.B, cl.C, cl.D, cl.dt)
