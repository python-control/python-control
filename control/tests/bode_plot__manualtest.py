'''
Created on 3 juli 2015

@author: marho40
'''

from __future__ import print_function
import scipy as sp
import matplotlib.pyplot as plt

import control

def test__bode_plot_PT1():
    testname = 'test__bode_plot_PT1'
    print ('\n\n{}{}\n'.format(testname, '-'*20))
    plt.figure()
    plt.gcf().suptitle(testname)

    w = (2*sp.pi*10.)  # 10 Hz        
    pt1 = control.tf([1.], [1/w, 1.])  # first order system
    print(pt1)
    pt1_bodeplot = control.bode_plot(pt1, omega=[2*sp.pi*1., 2*sp.pi*1000.], Plot=True, Hz=True, dB=True, color='b')
        

def test__bode_plot_PT1_sampled():
    testname = 'test__bode_plot_PT1_sampled'
    print ('\n\n{}{}\n'.format(testname, '-'*20))
    plt.figure()
    plt.gcf().suptitle(testname)
    
    sampleTime = 0.001 # 1ms    
    w = (2*sp.pi*10.)  # 10 Hz
    
    pt1 = control.tf([1.], [1/w, 1.])  # first order system
    print(pt1)
    pt1_bodeplot = control.bode_plot(pt1, omega=[2*sp.pi*1., 2*sp.pi*1000.], Plot=True, Hz=True, dB=True, color='b')
    
    pt1s = control.sample_system(pt1, sampleTime, 'tustin')
    print(pt1s)
    pt1s_bodeplot = control.bode_plot(pt1s, omega=[2*sp.pi*1., 2*sp.pi*500.], omega_num=1000, Plot=True, Hz=True, dB=True, color='r')


if __name__ == '__main__':
    test__bode_plot_PT1()
    test__bode_plot_PT1_sampled()
    plt.show()
    
    