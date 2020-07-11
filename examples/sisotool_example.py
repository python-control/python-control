#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sisotooldemo.py

Shows some different systems with sisotool. 

All should produce smooth root-locus plots, also zoomable and clickable, 
with proper branching
"""

#%%
import matplotlib.pyplot as plt
from control.matlab import *

# first example, aircraft attitude equation
s = tf([1,0],[1])
Kq = -24
T2 = 1.4
damping = 2/(13**.5)
omega = 13**.5
H = (Kq*(1+T2*s))/(s*(s**2+2*damping*omega*s+omega**2))
plt.close('all')
sisotool(-H)

#%%

# a simple RL, with multiple poles in the origin
plt.close('all')
H = (s+0.3)/(s**4 + 4*s**3 + 6.25*s**2)
sisotool(H)

#%%

# a branching and emanating example
b0 = 0.2
b1 = 0.1
b2 = 0.5
a0 = 2.3
a1 = 6.3
a2 = 3.6
a3 = 1.0

plt.close('all')
H = (b0 + b1*s + b2*s**2) / (a0 + a1*s + a2*s**2 + a3*s**3)

sisotool(H)
