__all__ = ['sisotool', 'rootlocus_pid_designer']

from control.exception import ControlMIMONotImplemented
from .freqplot import bode_plot
from .timeresp import step_response
from .lti import issiso, isdtime
from .xferfcn import tf
from .statesp import ss
from .bdalg import append, connect
from .iosys import tf2io, ss2io, summing_junction, interconnect
from control.statesp import _convert_to_statespace, StateSpace
from control.lti import common_timebase, isctime
import matplotlib
import matplotlib.pyplot as plt
import warnings

def sisotool(sys, kvect=None, xlim_rlocus=None, ylim_rlocus=None,
             plotstr_rlocus='C0', rlocus_grid=False, omega=None, dB=None,
             Hz=None, deg=None, omega_limits=None, omega_num=None,
             margins_bode=True, tvect=None):
    """
    Sisotool style collection of plots inspired by MATLAB's sisotool.
    The left two plots contain the bode magnitude and phase diagrams.
    The top right plot is a clickable root locus plot, clicking on the
    root locus will change the gain of the system. The bottom left plot
    shows a closed loop time response.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems. If sys is SISO, use the same
        system for the root locus and step response. If it is desired to
        see a different step response than feedback(K*loop,1), sys can be
        provided as a two-input, two-output system (e.g. by using
        :func:`bdgalg.connect' or :func:`iosys.interconnect`). Sisotool
        inserts the negative of the selected gain K between the first output
        and first input and uses the second input and output for computing
        the step response. This allows you to see the step responses of more
        complex systems, for example, systems with a feedforward path into the
        plant or in which the gain appears in the feedback path.
    kvect : list or ndarray, optional
        List of gains to use for plotting root locus
    xlim_rlocus : tuple or list, optional
        control of x-axis range, normally with tuple
        (see :doc:`matplotlib:api/axes_api`).
    ylim_rlocus : tuple or list, optional
        control of y-axis range
    plotstr_rlocus : :func:`matplotlib.pyplot.plot` format string, optional
        plotting style for the root locus plot(color, linestyle, etc)
    rlocus_grid : boolean (default = False)
        If True plot s- or z-plane grid.
    omega : array_like
        List of frequencies in rad/sec to be used for bode plot
    dB : boolean
        If True, plot result in dB for the bode plot
    Hz : boolean
        If True, plot frequency in Hz for the bode plot (omega must be provided in rad/sec)
    deg : boolean
        If True, plot phase in degrees for the bode plot (else radians)
    omega_limits : array_like of two values
        Limits of the to generate frequency vector.
        If Hz=True the limits are in Hz otherwise in rad/s. Ignored if omega
        is provided, and auto-generated if omitted.
    omega_num : int
        Number of samples to plot.  Defaults to
        config.defaults['freqplot.number_of_samples'].
    margins_bode : boolean
        If True, plot gain and phase margin in the bode plot
    tvect : list or ndarray, optional
        List of timesteps to use for closed loop step response

    Examples
    --------
    >>> sys = tf([1000], [1,25,100,0])
    >>> sisotool(sys)

    """
    from .rlocus import root_locus

    # sys as loop transfer function if SISO
    if not sys.issiso():
        if not (sys.ninputs == 2 and sys.noutputs == 2):
            raise ControlMIMONotImplemented(
                'sys must be SISO or 2-input, 2-output')

    # Setup sisotool figure or superimpose if one is already present
    fig = plt.gcf()
    if fig.canvas.manager.get_window_title() != 'Sisotool':
        plt.close(fig)
        fig,axes = plt.subplots(2, 2)
        fig.canvas.manager.set_window_title('Sisotool')

    # Extract bode plot parameters
    bode_plot_params = {
        'omega': omega,
        'dB': dB,
        'Hz': Hz,
        'deg': deg,
        'omega_limits': omega_limits,
        'omega_num' : omega_num,
        'sisotool': True,
        'fig': fig,
        'margins': margins_bode
    }

    # First time call to setup the bode and step response plots
    _SisotoolUpdate(sys, fig,
        1 if kvect is None else kvect[0], bode_plot_params)

    # Setup the root-locus plot window
    root_locus(sys, kvect=kvect, xlim=xlim_rlocus,
        ylim=ylim_rlocus, plotstr=plotstr_rlocus, grid=rlocus_grid,
        fig=fig, bode_plot_params=bode_plot_params, tvect=tvect, sisotool=True)

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

    # Update the bodeplot
    bode_plot_params['syslist'] = sys_loop*K.real
    bode_plot(**bode_plot_params)

    # Set the titles and labels
    ax_mag.set_title('Bode magnitude',fontsize = title_font_size)
    ax_mag.set_ylabel(ax_mag.get_ylabel(), fontsize=label_font_size)
    ax_mag.tick_params(axis='both', which='major', labelsize=label_font_size)

    ax_phase.set_title('Bode phase',fontsize=title_font_size)
    ax_phase.set_xlabel(ax_phase.get_xlabel(),fontsize=label_font_size)
    ax_phase.set_ylabel(ax_phase.get_ylabel(),fontsize=label_font_size)
    ax_phase.get_xaxis().set_label_coords(0.5, -0.15)
    ax_phase.get_shared_x_axes().join(ax_phase, ax_mag)
    ax_phase.tick_params(axis='both', which='major', labelsize=label_font_size)

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
                           Kp0=0, Ki0=0, Kd0=0, tau=0.01,
                           C_ff=0, derivative_in_feedback_path=False,
                           plot=True):
    """Manual PID controller design based on root locus using Sisotool

    Uses `Sisotool` to investigate the effect of adding or subtracting an
    amount `deltaK` to the proportional, integral, or derivative (PID) gains of
    a controller. One of the PID gains, `Kp`, `Ki`, or `Kd`, respectively, can
    be modified at a time. `Sisotool` plots the step response, frequency
    response, and root locus.

    When first run, `deltaK` is set to 0; click on a branch of the root locus
    plot to try a different value. Each click updates plots and prints
    the corresponding `deltaK`. To tune all three PID gains, repeatedly call
    `rootlocus_pid_designer`, and select a different `gain` each time (`'P'`,
    `'I'`, or `'D'`). Make sure to add the resulting `deltaK` to your chosen
    initial gain on the next iteration.

    Example: to examine the effect of varying `Kp` starting from an intial
    value of 10, use the arguments `gain='P', Kp0=10`. Suppose a `deltaK`
    value of 5 gives satisfactory performance. Then on the next iteration,
    to tune the derivative gain, use the arguments `gain='D', Kp0=15`.

    By default, all three PID terms are in the forward path C_f in the diagram
    shown below, that is,

    C_f = Kp + Ki/s + Kd*s/(tau*s + 1).

    If `plant` is a discrete-time system, then the proportional, integral, and
    derivative terms are given instead by Kp, Ki*dt/2*(z+1)/(z-1), and
    Kd/dt*(z-1)/z, respectively.

        ------> C_ff ------    d
        |                 |    |
    r   |     e           V    V  u         y
    ------->O---> C_f --->O--->O---> plant --->
            ^-            ^-                |
            |             |                 |
            |             ----- C_b <-------|
            ---------------------------------

    It is also possible to move the derivative term into the feedback path
    `C_b` using `derivative_in_feedback_path=True`. This may be desired to
    avoid that the plant is subject to an impulse function when the reference
    `r` is a step input. `C_b` is otherwise set to zero.

    If `plant` is a 2-input system, the disturbance `d` is fed directly into
    its second input rather than being added to `u`.

    Remark: It may be helpful to zoom in using the magnifying glass on the
    plot. Just ake sure to deactivate magnification mode when you are done by
    clicking the magnifying glass. Otherwise you will not be able to be able to choose
    a gain on the root locus plot.

    Parameters
    ----------
    plant : :class:`LTI` (:class:`TransferFunction` or :class:`StateSpace` system)
        The dynamical system to be controlled
    gain : string (optional)
        Which gain to vary by `deltaK`. Must be one of `'P'`, `'I'`, or `'D'`
        (proportional, integral, or derative)
    sign : int (optional)
        The sign of deltaK gain perturbation
    input : string (optional)
        The input used for the step response; must be `'r'` (reference) or
        `'d'` (disturbance) (see figure above)
    Kp0, Ki0, Kd0 : float (optional)
        Initial values for proportional, integral, and derivative gains,
        respectively
    tau : float (optional)
        The time constant associated with the pole in the continuous-time
        derivative term. This is required to make the derivative transfer
        function proper.
    C_ff : float or :class:`LTI` system (optional)
        Feedforward controller. If :class:`LTI`, must have timebase that is
        compatible with plant.
    derivative_in_feedback_path : bool (optional)
        Whether to place the derivative term in feedback transfer function
        `C_b` instead of the forward transfer function `C_f`.
    plot : bool (optional)
        Whether to create Sisotool interactive plot.

    Returns
    ----------
    closedloop : class:`StateSpace` system
        The closed-loop system using initial gains.
    """

    plant = _convert_to_statespace(plant)
    if plant.ninputs == 1:
        plant = ss2io(plant, inputs='u', outputs='y')
    elif plant.ninputs == 2:
        plant = ss2io(plant, inputs=['u', 'd'], outputs='y')
    else:
        raise ValueError("plant must have one or two inputs")
    C_ff = ss2io(_convert_to_statespace(C_ff),   inputs='r', outputs='uff')
    dt = common_timebase(plant, C_ff)

    # create systems used for interconnections
    e_summer = summing_junction(['r', '-y'], 'e')
    if plant.ninputs == 2:
        u_summer = summing_junction(['ufb', 'uff'], 'u')
    else:
        u_summer = summing_junction(['ufb', 'uff', 'd'], 'u')

    if isctime(plant):
        prop  = tf(1, 1)
        integ = tf(1, [1, 0])
        deriv = tf([1, 0], [tau, 1])
    else: # discrete-time
        prop  = tf(1, 1, dt)
        integ = tf([dt/2, dt/2], [1, -1], dt)
        deriv = tf([1, -1], [dt, 0], dt)

    # add signal names by turning into iosystems
    prop  = tf2io(prop,        inputs='e', outputs='prop_e')
    integ = tf2io(integ,       inputs='e', outputs='int_e')
    if derivative_in_feedback_path:
        deriv = tf2io(-deriv,  inputs='y', outputs='deriv')
    else:
        deriv = tf2io(deriv,   inputs='e', outputs='deriv')

    # create gain blocks
    Kpgain = tf2io(tf(Kp0, 1),            inputs='prop_e',  outputs='ufb')
    Kigain = tf2io(tf(Ki0, 1),            inputs='int_e',   outputs='ufb')
    Kdgain = tf2io(tf(Kd0, 1),            inputs='deriv',  outputs='ufb')

    # for the gain that is varied, replace gain block with a special block
    # that has an 'input' and an 'output' that creates loop transfer function
    if gain in ('P', 'p'):
        Kpgain = ss2io(ss([],[],[],[[0, 1], [-sign, Kp0]]),
            inputs=['input', 'prop_e'], outputs=['output', 'ufb'])
    elif gain in ('I', 'i'):
        Kigain = ss2io(ss([],[],[],[[0, 1], [-sign, Ki0]]),
            inputs=['input', 'int_e'],  outputs=['output', 'ufb'])
    elif gain in ('D', 'd'):
        Kdgain = ss2io(ss([],[],[],[[0, 1], [-sign, Kd0]]),
            inputs=['input', 'deriv'], outputs=['output', 'ufb'])
    else:
        raise ValueError(gain + ' gain not recognized.')

    # the second input and output are used by sisotool to plot step response
    loop = interconnect((plant, Kpgain, Kigain, Kdgain, prop, integ, deriv,
                            C_ff, e_summer, u_summer),
                            inplist=['input', input_signal],
                            outlist=['output', 'y'], check_unused=False)
    if plot:
        sisotool(loop, kvect=(0.,))
    cl = loop[1, 1] # closed loop transfer function with initial gains
    return StateSpace(cl.A, cl.B, cl.C, cl.D, cl.dt)
