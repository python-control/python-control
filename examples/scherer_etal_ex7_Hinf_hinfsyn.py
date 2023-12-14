"""Hinf design using hinfsyn.

Demonstrate Hinf design for a SISO plant using hinfsyn. Based on [1], Ex. 7.

[1] Scherer, Gahinet, & Chilali, "Multiobjective Output-Feedback Control via
LMI Optimization", IEEE Trans. Automatic Control, Vol. 42, No. 7, July 1997.
"""
# %%
# Packages
import numpy as np
import control

# %%
# State-space system.

# Process model.
A = np.array([[0, 10, 2],
              [-1, 1, 0],
              [0, 2, -5]])
B1 = np.array([[1],
               [0],
               [1]])
B2 = np.array([[0],
               [1],
               [0]])

# Plant output.
C2 = np.array([[0, 1, 0]])
D21 = np.array([[2]])
D22 = np.array([[0]])

# Hinf performance.
C1 = np.array([[1, 0, 0],
               [0, 0, 0]])
D11 = np.array([[0],
                [0]])
D12 = np.array([[0],
                [1]])

# Dimensions.
n_x, n_u, n_y = 3, 1, 1

# %%
# Hinf design using hinfsyn.

# Create augmented plant.
Baug = np.block([B1, B2])
Caug = np.block([[C1], [C2]])
Daug = np.block([[D11, D12], [D21, D22]])
Paug = control.ss(A, Baug, Caug, Daug)

# Input to hinfsyn is Paug, number of inputs to controller,
# and number of outputs from the controller.
K, Tzw, gamma, rcond = control.hinfsyn(Paug, n_y, n_u)
print(f'The closed-loop H_inf norm of Tzw(s) is {gamma}.')

# %%
