# mkd_era.py
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
response = ct.impulse_response(sysd)
response.plot()
plt.show()

sysd_est, _ = ct.era(response,r=4,dt=dt)

step_true = ct.step_response(sysd)
step_est = ct.step_response(sysd_est)

step_true.plot(title=xixo)
step_est.plot(color='orange',linestyle='dashed')

plt.show()


if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:

    plt.show()