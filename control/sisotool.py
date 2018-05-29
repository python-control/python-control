__all__ = ['sisotool']

from .freqplot import bode_plot
from .timeresp import step_response
from .lti import issiso
import matplotlib
import matplotlib.pyplot as plt

def sisotool(sys, kvect = None, xlim = None, ylim = None, plotstr_rlocus = 'b' if int(matplotlib.__version__[0]) == 1 else 'C0',rlocus_grid = False, omega = None, dB = None, Hz = None, deg = None, omega_limits = None, omega_num = None,margins = True, tvect=None):

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
        'margins': margins
    }

    # First time call to setup the bode and step response plots
    _SisotoolUpdate(sys, fig,1 if kvect is None else kvect[0],bode_plot_params)

    # Setup the root-locus plot window
    root_locus(sys,kvect=kvect,xlim=xlim,ylim = ylim,plotstr=plotstr_rlocus,grid = rlocus_grid,fig=fig,bode_plot_params=bode_plot_params,tvect=tvect,sisotool=True)

def _SisotoolUpdate(sys,fig,K,bode_plot_params,tvect=None):

    if int(matplotlib.__version__[0]) == 1:
        title_font_size = 12
        label_font_size = 10
    else:
        title_font_size = 10
        label_font_size = 8

    # Get the subaxes and clear them
    ax_mag,ax_rlocus,ax_phase,ax_step = fig.axes[0],fig.axes[1],fig.axes[2],fig.axes[3]
    ax_mag.cla(),ax_phase.cla(),ax_step.cla()

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

