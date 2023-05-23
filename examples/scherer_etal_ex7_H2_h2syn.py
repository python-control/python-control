"""H2 design using h2syn.

Demonstrate H2 desing for a SISO plant using h2syn. Based on [1], Ex. 7.

[1] Scherer, Gahinet, & Chilali, "Multiobjective Output-Feedback Control via
LMI Optimization", IEEE Trans. Automatic Control, Vol. 42, No. 7, July 1997.

[2] Zhou & Doyle, "Essentials of Robust Control", Prentice Hall,
Upper Saddle River, NJ, 1998.
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

# H2 performance.
C1 = np.array([[0, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])
D11 = np.array([[0],
                [0],
                [0]])
D12 = np.array([[0],
                [0],
                [1]])

# Dimensions.
n_u, n_y = 1, 1

# %%
# H2 design using h2syn.

# Create augmented plant.
Baug = np.block([B1, B2])
Caug = np.block([[C1], [C2]])
Daug = np.block([[D11, D12], [D21, D22]])
Paug = control.ss(A, Baug, Caug, Daug)

# Input to h2syn is Paug, number of inputs to controller,
# and number of outputs from the controller.
K = control.h2syn(Paug, n_y, n_u)

# Extarct controller ss realization.
A_K, B_K, C_K, D_K = K.A, K.B, K.C, K.D

# %%
# Compute closed-loop H2 norm.

# Compute closed-loop system, Tzw(s). See Eq. 4 in [1].
Azw = np.block([[A + B2 @ D_K @ C2, B2 @ C_K],
                [B_K @ C2, A_K]])
Bzw = np.block([[B1 + B2 @ D_K @ D21],
                [B_K @ D21]])
Czw = np.block([C1 + D12 @ D_K @ C2, D12 @ C_K])
Dzw = D11 + D12 @ D_K @ D21
Tzw = control.ss(Azw, Bzw, Czw, Dzw)

# Compute closed-loop H2 norm via Lyapunov equation.
# See [2], Lemma 4.4, pg 53.
Qzw = control.lyap(Azw.T, Czw.T @ Czw)
nu = np.sqrt(np.trace(Bzw.T @ Qzw @ Bzw))
print(f'The closed-loop H_2 norm of Tzw(s) is {nu}.')
# Value is 7.748350599360575, the same as reported in [1].

# %%
