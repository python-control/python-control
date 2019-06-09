__all__ = ['sisotool']

from .freqplot import bode_plot
from .timeresp import step_response
from .lti import issiso
import matplotlib
import matplotlib.pyplot as plt
import warnings

def sisotool(sys, kvect = None, xlim_rlocus = None, ylim_rlocus = None,
             plotstr_rlocus = 'b' if int(matplotlib.__version__[0]) == 1 else 'C0',
             rlocus_grid = False, omega = None, dB = None, Hz = None,
             deg = None, omega_limits = None, omega_num = None,
             margins_bode = True, tvect=None):
    """
    Sisotool style collection of plots inspired by MATLAB's sisotool.
    The left two plots contain the bode magnitude and phase diagrams.
    The top right plot is a clickable root locus plot, clicking on the
    root locus will change the gain of the system. The bottom left plot
    shows a closed loop time response.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems (SISO only)
    kvect : list or ndarray, optional
        List of gains to use for plotting root locus
    xlim_rlocus : tuple or list, optional
        control of x-axis range, normally with tuple (see matplotlib.axes)
    ylim_rlocus : tuple or list, optional
        control of y-axis range
    plotstr_rlocus : Additional options to matplotlib
        plotting style for the root locus plot(color, linestyle, etc)
    rlocus_grid: boolean (default = False)
        If True plot s-plane grid.
    omega : freq_range
        Range of frequencies in rad/sec for the bode plot
    dB : boolean
        If True, plot result in dB for the bode plot
    Hz : boolean
        If True, plot frequency in Hz for the bode plot (omega must be provided in rad/sec)
    deg : boolean
        If True, plot phase in degrees for the bode plot (else radians)
    omega_limits: tuple, list, ... of two values
        Limits of the to generate frequency vector.
        If Hz=True the limits are in Hz otherwise in rad/s.
    omega_num: int
        number of samples
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

    # Check if it is a single SISO system
    issiso(sys,strict=True)

    # Setup sisotool figure or superimpose if one is already present
    fig = plt.gcf()
    if fig.canvas.get_window_title() != 'Sisotool':
        plt.close(fig)
        fig,axes = plt.subplots(2, 2)
        fig.canvas.set_window_title('Sisotool')

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
    _SisotoolUpdate(sys, fig,1 if kvect is None else kvect[0],bode_plot_params)

    # Setup the root-locus plot window
    root_locus(sys,kvect=kvect,xlim=xlim_rlocus,ylim = ylim_rlocus,plotstr=plotstr_rlocus,grid = rlocus_grid,fig=fig,bode_plot_params=bode_plot_params,tvect=tvect,sisotool=True)

def _SisotoolUpdate(sys,fig,K,bode_plot_params,tvect=None):

    if int(matplotlib.__version__[0]) == 1:
        title_font_size = 12
        label_font_size = 10
    else:
        title_font_size = 10
        label_font_size = 8

    # Get the subaxes and clear them
    ax_mag,ax_rlocus,ax_phase,ax_step = fig.axes[0],fig.axes[1],fig.axes[2],fig.axes[3]

    # Catch matplotlib 2.1.x and higher userwarnings when clearing a log axis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax_step.clear(), ax_mag.clear(), ax_phase.clear()

    # Update the bodeplot
    bode_plot_params['syslist'] = sys*K.real
    bode_plot(**bode_plot_params)

    # Set the titles and labels
    ax_mag.set_title('Bode magnitude',fontsize = title_font_size)
    ax_mag.set_ylabel(ax_mag.get_ylabel(), fontsize=label_font_size)

    ax_phase.set_title('Bode phase',fontsize=title_font_size)
    ax_phase.set_xlabel(ax_phase.get_xlabel(),fontsize=label_font_size)
    ax_phase.set_ylabel(ax_phase.get_ylabel(),fontsize=label_font_size)
    ax_phase.get_xaxis().set_label_coords(0.5, -0.15)
    ax_phase.get_shared_x_axes().join(ax_phase, ax_mag)

    ax_step.set_title('Step response',fontsize = title_font_size)
    ax_step.set_xlabel('Time (seconds)',fontsize=label_font_size)
    ax_step.set_ylabel('Amplitude',fontsize=label_font_size)
    ax_step.get_xaxis().set_label_coords(0.5, -0.15)
    ax_step.get_yaxis().set_label_coords(-0.15, 0.5)

    ax_rlocus.set_title('Root locus',fontsize = title_font_size)
    ax_rlocus.set_ylabel('Imag', fontsize=label_font_size)
    ax_rlocus.set_xlabel('Real', fontsize=label_font_size)
    ax_rlocus.get_xaxis().set_label_coords(0.5, -0.15)
    ax_rlocus.get_yaxis().set_label_coords(-0.15, 0.5)



    # Generate the step response and plot it
    sys_closed = (K*sys).feedback(1)
    if tvect is None:
        tvect, yout = step_response(sys_closed)
    else:
        tvect, yout = step_response(sys_closed,tvect)
    ax_step.plot(tvect, yout)
    ax_step.axhline(1.,linestyle=':',color='k',zorder=-20)

    # Manually adjust the spacing and draw the canvas
    fig.subplots_adjust(top=0.9,wspace = 0.3,hspace=0.35)
    fig.canvas.draw()

