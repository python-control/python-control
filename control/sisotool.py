__all__ = ['sisotool']

from .freqplot import bode_plot
from .timeresp import step_response
from .lti import issiso
import matplotlib.pyplot as plt

def sisotool(sys, kvect = None, xlim = None, ylim = None, plotstr_rlocus = '-',rlocus_grid = False, omega = None, dB = None, Hz = None, deg = None, omega_limits = None, omega_num = None, tvect=None):

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
    }

    # First time call to setup the bode and step response plots
    _SisotoolUpdate(sys, fig,1 if kvect is None else kvect[0],bode_plot_params)

    # Setup the root-locus plot window
    root_locus(sys,kvect=kvect,xlim=xlim,ylim = ylim,plotstr=plotstr_rlocus,grid = rlocus_grid,fig=fig,bode_plot_params=bode_plot_params,tvect=tvect,sisotool=True)

def _SisotoolUpdate(sys,fig,K,bode_plot_params,tvect=None):

    # Get the subaxes and clear them
    ax_mag,ax_rlocus,ax_phase,ax_step = fig.axes[0],fig.axes[1],fig.axes[2],fig.axes[3]
    ax_mag.cla(),ax_phase.cla(),ax_step.cla()

    # Set the titles and labels
    ax_mag.set_title('Bode magnitude')
    ax_phase.set_title('Bode phase')
    ax_rlocus.set_title('Root locus')
    ax_step.set_title('Step response')
    ax_step.set_xlabel('Time (seconds)')
    ax_step.set_ylabel('Amplitude')

    # Update the bodeplot
    bode_plot_params['syslist'] = sys*K.real
    bode_plot(**bode_plot_params)

    # Generate the step response and plot it
    sys_closed = (K*sys).feedback(1)
    if tvect is None:
        tvect, yout = step_response(sys_closed)
    else:
        tvect, yout = step_response(sys_closed,tvect)
    ax_step.plot(tvect, yout)
    ax_step.axhline(1.,linestyle=':',color='k')

