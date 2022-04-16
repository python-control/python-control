#!/usr/bin/python
# needs pmw (in pypi, conda-forge)

""" Simple GUI application for visualizing how the poles/zeros of the transfer
function effects the bode, nyquist and step response of a SISO system """

"""Copyright (c) 2011, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the project author nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Vanessa Romero Segovia
Author: Ola Johnsson
Author: Jerker Nordh
"""

import control.matlab
import tkinter
import sys
import Pmw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.lib.polynomial import polymul
from numpy.lib.type_check import real
from numpy.core.multiarray import array
from numpy.core.fromnumeric import size
# from numpy.lib.function_base import logspace
from control.matlab import logspace
from numpy import conj


def make_poly(facts):
    """ Create polynomial from factors """
    poly = [1]
    for factor in facts:
        poly = polymul(poly, [1, -factor])

    return real(poly)


def coeff_string_check(text):
    """ Check so textfield entry is valid string of coeffs. """
    try:
        [float(a) for a in text.split()]
    except:
        return Pmw.PARTIAL

    return Pmw.OK


class TFInput:
    """ Class for handling input of transfer function coeffs."""
    def __init__(self, parent):
        self.master = parent
        self.denominator = []
        self.numerator = []
        self.numerator_widget = Pmw.EntryField(self.master,
                                    labelpos='w',
                                    label_text='Numerator',
                                    entry_width = 25,
                                    validate=coeff_string_check,
                                    value='1.0 -6.0 12.0')
        self.denominator_widget = Pmw.EntryField(self.master,
                                    labelpos='w',
                                    label_text='Denominator',
                                    entry_width = 25,
                                    validate=coeff_string_check,
                                    value='1.0 5.0 14.0 27.0')
        self.balloon = Pmw.Balloon(self.master)

        try:
            self.balloon.bind(self.numerator_widget,
                                "Numerator coefficients, e.g: 1.0 2.0")
        except:
            pass

        try:
            self.balloon.bind(self.denominator_widget,
                                "Denominator coefficients, e.g: 1.0 3.0 2.0")
        except:
            pass

        widgets = (self.numerator_widget, self.denominator_widget)
        for i in range(len(widgets)):
            widgets[i].grid(row=i+1, column=0, padx=20, pady=3)
        Pmw.alignlabels(widgets)

        self.numerator_widget.component('entry').focus_set()

    def get_tf(self):
        """ Return transfer functions object created from coeffs"""
        try:
            numerator = (
                [float(a) for a in self.numerator_widget.get().split()])
        except:
            numerator = None

        try:
            denominator = (
                [float(a) for a in self.denominator_widget.get().split()])
        except:
            denominator = None

        try:
            if (numerator != None and denominator != None):
                tfcn = control.matlab.tf(numerator, denominator)
            else:
                tfcn = None
        except:
            tfcn = None

        return tfcn



    def set_poles(self, poles):
        """ Set the poles to the new positions"""
        self.denominator = make_poly(poles)
        self.denominator_widget.setentry(
            ' '.join([format(i,'.3g') for i in self.denominator]))

    def set_zeros(self, zeros):
        """ Set the zeros to the new positions"""
        self.numerator = make_poly(zeros)
        self.numerator_widget.setentry(
            ' '.join([format(i,'.3g') for i in self.numerator]))


class Analysis:
    """ Main class for GUI visualising transfer functions """
    def __init__(self, parent):
        """Creates all widgets"""
        self.master = parent
        self.move_zero = None
        self.index1 = None
        self.index2 = None
        self.zeros = []
        self.poles = []

        self.topframe = tkinter.Frame(self.master)
        self.topframe.pack(expand=True, fill='both')

        self.entries = tkinter.Frame(self.topframe)
        self.entries.pack(expand=True, fill='both')

        self.figure = tkinter.Frame(self.topframe)
        self.figure.pack(expand=True, fill='both')

        header = tkinter.Label(self.entries,
            text='Define the transfer function:')
        header.grid(row=0, column=0, padx=20, pady=7)


        self.tfi = TFInput(self.entries)
        self.sys = self.tfi.get_tf()

        tkinter.Button(self.entries, text='Apply', command=self.apply,
                       width=9).grid(row=0, column=1, rowspan=3, padx=10, pady=5)

        self.f_bode = plt.figure(figsize=(4, 4))
        self.f_nyquist = plt.figure(figsize=(4, 4))
        self.f_pzmap = plt.figure(figsize=(4, 4))
        self.f_step = plt.figure(figsize=(4, 4))

        self.canvas_pzmap = FigureCanvasTkAgg(self.f_pzmap,
                                              master=self.figure)
        self.canvas_pzmap.draw()
        self.canvas_pzmap.get_tk_widget().grid(row=0, column=0,
                                               padx=0, pady=0)

        self.canvas_bode = FigureCanvasTkAgg(self.f_bode,
                                             master=self.figure)
        self.canvas_bode.draw()
        self.canvas_bode.get_tk_widget().grid(row=0, column=1,
                                              padx=0, pady=0)

        self.canvas_step = FigureCanvasTkAgg(self.f_step,
                                             master=self.figure)
        self.canvas_step.draw()
        self.canvas_step.get_tk_widget().grid(row=1, column=0,
                                              padx=0, pady=0)

        self.canvas_nyquist = FigureCanvasTkAgg(self.f_nyquist,
                                                master=self.figure)
        self.canvas_nyquist.draw()
        self.canvas_nyquist.get_tk_widget().grid(row=1, column=1,
                                                 padx=0, pady=0)

        self.canvas_pzmap.mpl_connect('button_press_event',
                                      self.button_press)
        self.canvas_pzmap.mpl_connect('button_release_event',
                                      self.button_release)
        self.canvas_pzmap.mpl_connect('motion_notify_event',
                                      self.mouse_move)

        self.apply()

    def button_press(self, event):
        """ Handle button presses, detect if we are going to move
        any poles/zeros"""
        # find closest pole/zero
        if event.xdata != None and event.ydata != None:

            new = event.xdata + 1.0j*event.ydata

            tzeros = list(abs(self.zeros-new))
            tpoles = list(abs(self.poles-new))

            if (size(tzeros) > 0):
                minz = min(tzeros)
            else:
                minz = float('inf')
            if (size(tpoles) > 0):
                minp = min(tpoles)
            else:
                minp = float('inf')

            if (minz < 2 or minp < 2):
                if (minz < minp):
                    # Moving zero(s)
                    self.index1 = tzeros.index(minz)
                    self.index2 = list(self.zeros).index(
                        conj(self.zeros[self.index1]))
                    self.move_zero = True
                else:
                    # Moving pole(s)
                    self.index1 = tpoles.index(minp)
                    self.index2 = list(self.poles).index(
                        conj(self.poles[self.index1]))
                    self.move_zero = False

    def button_release(self, event):
        """ Handle button release, update pole/zero positions,
        if the were moved"""
        if (self.move_zero == True):
            self.tfi.set_zeros(self.zeros)
        elif (self.move_zero == False):
            self.tfi.set_poles(self.poles)
        else:
            return

        self.move_zero = None
        self.index1 = None
        self.index2 = None

        tfcn = self.tfi.get_tf()
        if (tfcn):
            self.zeros = tfcn.zeros()
            self.poles = tfcn.poles()
            self.sys = tfcn
            self.redraw()

    def mouse_move(self, event):
        """ Handle mouse movement, redraw pzmap while drag/dropping """
        if (self.move_zero != None and
            event.xdata != None and
            event.ydata != None):

            if (self.index1 == self.index2):
                # Real pole/zero
                new = event.xdata
                if (self.move_zero == True):
                    self.zeros[self.index1] = new
                elif (self.move_zero == False):
                    self.poles[self.index1] = new
            else:
                # Complex poles/zeros
                new = event.xdata + 1.0j*event.ydata
                if (self.move_zero == True):
                    self.zeros[self.index1] = new
                    self.zeros[self.index2] = conj(new)
                elif (self.move_zero == False):
                    self.poles[self.index1] = new
                    self.poles[self.index2] = conj(new)
            tfcn = None
            if (self.move_zero == True):
                self.tfi.set_zeros(self.zeros)
                tfcn = self.tfi.get_tf()
            elif (self.move_zero == False):
                self.tfi.set_poles(self.poles)
                tfcn = self.tfi.get_tf()
            if (tfcn != None):
                self.draw_pz(tfcn)
                self.canvas_pzmap.draw()

    def apply(self):
        """Evaluates the transfer function and produces different plots for
           analysis"""
        tfcn = self.tfi.get_tf()

        if (tfcn):
            self.zeros = tfcn.zeros()
            self.poles = tfcn.poles()
            self.sys = tfcn
            self.redraw()

    def draw_pz(self, tfcn):
        """Draw pzmap"""
        self.f_pzmap.clf()
        # Make adaptive window size, with min [-10, 10] in range,
        # always atleast 25% extra space outside poles/zeros
        tmp = list(self.zeros)+list(self.poles)+[8]
        val = 1.25*max(abs(array(tmp)))
        plt.figure(self.f_pzmap.number)
        control.matlab.pzmap(tfcn)
        plt.suptitle('Pole-Zero Diagram')

        plt.axis([-val, val, -val, val])

    def redraw(self):
        """ Redraw all diagrams """
        self.draw_pz(self.sys)

        self.f_bode.clf()
        plt.figure(self.f_bode.number)
        control.matlab.bode(self.sys, logspace(-2, 2, 1000))
        plt.suptitle('Bode Diagram')

        self.f_nyquist.clf()
        plt.figure(self.f_nyquist.number)
        control.matlab.nyquist(self.sys, logspace(-2, 2, 1000))
        plt.suptitle('Nyquist Diagram')

        self.f_step.clf()
        plt.figure(self.f_step.number)
        try:
            # Step seems to get intro trouble
            # with purely imaginary poles
            yvec, tvec = control.matlab.step(self.sys)
            plt.plot(tvec.T, yvec)
        except:
            print("Error plotting step response")
        plt.suptitle('Step Response')

        self.canvas_pzmap.draw()
        self.canvas_bode.draw()
        self.canvas_step.draw()
        self.canvas_nyquist.draw()


def create_analysis():
    """ Create main object """
    def handler():
        """ Handles WM_DELETE_WINDOW messages """
        root.destroy()
        sys.exit()

    # Launch a GUI for the Analysis module
    root = tkinter.Tk()
    root.protocol("WM_DELETE_WINDOW", handler)
    Pmw.initialise(root)
    root.title('Analysis of Linear Systems')
    Analysis(root)
    root.mainloop()


if __name__ == '__main__':
    import os
    if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
        create_analysis()
