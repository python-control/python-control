# markov.py
# Johannes Kaisinger, 4 July 2024
#
# Demonstrate estimation of markov parameters.
# SISO, SIMO, MISO, MIMO case

import numpy as np
import matplotlib.pyplot as plt
import os

import control as ct

# set up a mass spring damper system (2dof, MIMO case)
# m q_dd + c q_d + k q = u
m1, k1, c1 = 1., 1., .1
m2, k2, c2 = 2., .5, .1
k3, c3 = .5, .1

A = np.array([
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [-(k1+k2)/m1, (k2)/m1, -(c1+c2)/m1, c2/m1],
    [(k2)/m2, -(k2+k3)/m2, c2/m2, -(c2+c3)/m2]
])
B = np.array([[0.,0.],[0.,0.],[1/m1,0.],[0.,1/m2]])
C = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0]])
D = np.zeros((2,2))


xixo_list = ["SISO","SIMO","MISO","MIMO"]
xixo = xixo_list[3] # choose a system for estimation
match xixo:
    case "SISO":
        sys = ct.StateSpace(A, B[:,0], C[0,:], D[0,0])
    case "SIMO":
        sys = ct.StateSpace(A, B[:,:1], C, D[:,:1])
    case "MISO":
        sys = ct.StateSpace(A, B, C[:1,:], D[:1,:])
    case "MIMO":
        sys = ct.StateSpace(A, B, C, D)

dt = 0.5
sysd = sys.sample(dt, method='zoh')

t = np.arange(0,5000,dt)
u = np.random.randn(sysd.B.shape[-1], len(t)) # random forcing input

response = ct.forced_response(sysd, U=u)
response.plot()
plt.show()

markov_true = ct.impulse_response(sysd,T=dt*100)
markov_est = ct.markov(response,m=100,dt=dt)

markov_true.plot(title=xixo)
markov_est.plot(color='orange',linestyle='dashed')
plt.show()

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()