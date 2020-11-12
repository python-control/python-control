from matplotlib import pyplot as plt

import numpy as np
from scipy.linalg import eig
from control import step_response, StateSpace
# Example of diagonalization of StateSpace representation
# In this example the diagonalized system has complex values on the matrices

def diagonalize(A, B, C, D):
    eigenvectors_matrix = eig(A)[1]
    A = np.matmul(np.matmul(np.linalg.inv(eigenvectors_matrix), A), eigenvectors_matrix)
    B = np.matmul(np.linalg.inv(eigenvectors_matrix), B)
    C = np.matmul(C, eigenvectors_matrix)
    print(A)
    print(B)
    print(C)
    print(D)
    return A, B, C, D


def main():
    A = np.array([[0, 0, 1000], [0.1, 0, 0], [0, 0.5, 0.5]])
    B = np.array([[0], [0], [1]])
    C = np.array([[0, 0, 1]])
    D = np.array([[0]])

    Aw, Bw, Cw, Dw = diagonalize(A, B, C, D)

    system = StateSpace(A, B, C, D, 1)
    system_w = StateSpace(Aw, Bw, Cw, Dw, 1)

    length = 20
    plt.figure(figsize=(14, 9), dpi=100)
    response_w = step_response(system_w, T=np.arange(length))
    response = step_response(system, T=np.arange(length))
    plt.plot(response[0], response[1], color="red", label="System")
    plt.plot(response_w[0], response_w[1], color="blue", label="System W")
    plt.xticks(np.arange(length))
    plt.legend()
    plt.savefig("diagonalization.png")

main()