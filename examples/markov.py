# markov.py
# Johannes Kaisinger, 4 July 2024
#
# Demonstrate estimation of markov parameters.
# SISO, SIMO, MISO, MIMO case

import numpy as np
import matplotlib.pyplot as plt
import os

import control as ct

def create_impulse_response(H, time, transpose, dt):
    """Helper function to use TimeResponseData type for plotting"""

    H = np.array(H, ndmin=3)

    if transpose:
        H = np.transpose(H)

    q, p, m = H.shape
    inputs = np.zeros((p,p,m))

    issiso = True if (q == 1 and p == 1) else False

    input_labels = []
    trace_labels, trace_types = [], []
    for i in range(p):
        inputs[i,i,0] = 1/dt # unit area impulse
        input_labels.append(f"u{[i]}")
        trace_labels.append(f"From u{[i]}")
        trace_types.append('impulse')

    output_labels = []
    for i in range(q):
        output_labels.append(f"y{[i]}")

    return ct.TimeResponseData(time=time[:m],
                            outputs=H,
                            output_labels=output_labels,
                            inputs=inputs,
                            input_labels=input_labels,
                            trace_labels=trace_labels,
                            trace_types=trace_types,
                            sysname="H_est",
                            transpose=transpose,
                            plot_inputs=False,
                            issiso=issiso)

# set up a mass spring damper system (2dof, MIMO case)
# Mechanical Vibrations: Theory and Application, SI Edition, 1st ed.
# Figure 6.5 / Example 6.7
# m q_dd + c q_d + k q = f
m1, k1, c1 = 1., 4., 1.
m2, k2, c2 = 2., 2., 1.
k3, c3 = 6., 2.

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

dt = 0.25
sysd = sys.sample(dt, method='zoh')
sysd.name = "H_true"

 # random forcing input
t = np.arange(0,100,dt)
u = np.random.randn(sysd.B.shape[-1], len(t))

response = ct.forced_response(sysd, U=u)
response.plot()
plt.show()

m = 50
ir_true = ct.impulse_response(sysd, T=dt*m)

H_est = ct.markov(response, m, dt=dt)
# Helper function for plotting only
ir_est = create_impulse_response(H_est,
                                 ir_true.time,
                                 ir_true.transpose,
                                 dt)

ir_true.plot(title=xixo)
ir_est.plot(color='orange',linestyle='dashed')
plt.show()

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
