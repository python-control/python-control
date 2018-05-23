__all__ = ['sisotool']

from .freqplot import bode_plot
from .rlocus import root_locus
import matplotlib.pyplot as plt

def sisotool(sys, kvect=None,PrintGain=True,grid=False,dB=None,Hz=None,deg=None):
    f, (ax1, ax2) = plt.subplots(1, 2)
    root_locus(sys,ax=ax1,f=f,sisotool=True)
    #bode_plot(sys,ax=ax2)