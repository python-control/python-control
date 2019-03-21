import numpy as np
import control


#unit test for stabilizing and anti-stabilizing feedbacks
#continuous-time

A = np.diag([1,-1])
B = np.identity(2)
Q = np.identity(2)
R = np.identity(2)
S = 0 * B
E = np.identity(2)
X, L , G = control.care(A, B, Q, R, S, E, stabilizing=True)
assert np.all(np.real(L) < 0)
X, L , G = control.care(A, B, Q, R, S, E, stabilizing=False)
assert np.all(np.real(L) > 0)


#discrete-time
A = np.diag([0.5,2])
B = np.identity(2)
Q = np.identity(2)
R = np.identity(2)
S = 0 * B
E = np.identity(2)
X, L , G = control.dare(A, B, Q, R, S, E, stabilizing=True)
assert np.all(np.abs(L) < 1)
X, L , G = control.dare(A, B, Q, R, S, E, stabilizing=False)
assert np.all(np.abs(L) > 1)
