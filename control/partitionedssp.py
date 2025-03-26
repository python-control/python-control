import numpy as np
from scipy.linalg import block_diag, inv

from .statesp import ss, StateSpace

class PartitionedStateSpace:
    def __init__(self, sys: StateSpace, nu1: int, ny1: int):
        self.sys = sys
        self.nu1 = nu1
        self.ny1 = ny1

        self.A = self.sys.A
        self.B = self.sys.B
        self.C = self.sys.C
        self.D = self.sys.D

    @property
    def B1(self):
        return self.B[:, :self.nu1]

    @property
    def B2(self):
        return self.B[:, self.nu1:]

    @property
    def C1(self):
        return self.C[:self.ny1, :]

    @property
    def C2(self):
        return self.C[self.ny1:, :]

    @property
    def D11(self):
        return self.D[:self.ny1, :self.nu1]

    @property
    def D12(self):
        return self.D[:self.ny1, self.nu1:]

    @property
    def D21(self):
        return self.D[self.ny1:, :self.nu1]

    @property
    def D22(self):
        return self.D[self.ny1:, self.nu1:]
    

    @classmethod
    def from_matrices(cls, A, B1, B2, C1, C2, D11, D12, D21, D22):
        nx = A.shape[0]   
        nw = B1.shape[1] 
        nu = B2.shape[1]  
        nz = C1.shape[0] 
        ny = C2.shape[0]

        # Shape validations
        if A.shape[1] != nx and nx != 0:
            raise ValueError("A must be square")
        if B1.shape[0] != nx:
            raise ValueError("B1 must have the same row size as A")
        if B2.shape[0] != nx:
            raise ValueError("B2 must have the same row size as A")
        if C1.shape[1] != nx:
            raise ValueError("C1 must have the same column size as A")
        if C2.shape[1] != nx:
            raise ValueError("C2 must have the same column size as A")
        if D11.shape[1] != nw:
            raise ValueError("D11 must have the same column size as B1")
        if D21.shape[1] != nw:
            raise ValueError("D21 must have the same column size as B1")
        if D12.shape[1] != nu:
            raise ValueError("D12 must have the same column size as B2")
        if D22.shape[1] != nu:
            raise ValueError("D22 must have the same column size as B2")
        if D11.shape[0] != nz:
            raise ValueError("D11 must have the same row size as C1")
        if D12.shape[0] != nz:
            raise ValueError("D12 must have the same row size as C1")
        if D21.shape[0] != ny:
            raise ValueError("D21 must have the same row size as C2")
        if D22.shape[0] != ny:
            raise ValueError("D22 must have the same row size as C2")

        B = np.hstack((B1, B2))
        C = np.vstack((C1, C2))
        D = np.block([[D11, D12], [D21, D22]])

        sys = ss(A, B, C, D)

        return cls(sys, nw, nz)
    

    def __add__(self, other):
        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Can only add PartitionedStateSpace objects")

        A = block_diag(self.A, other.A)
        B1 = np.vstack((self.B1, other.B1))
        B2 = block_diag(self.B2, other.B2)
        B = np.hstack((B1, B2))

        C1 = np.hstack((self.C1, other.C1))
        C2 = block_diag(self.C2, other.C2)
        C = np.vstack((C1, C2))

        D11 = self.D11 + other.D11
        D12 = np.hstack((self.D12, other.D12))
        D21 = np.vstack((self.D21, other.D21))
        D22 = block_diag(self.D22, other.D22)
        D = np.block([[D11, D12], [D21, D22]])

        P = ss(A, B, C, D)
        return PartitionedStateSpace(P, self.nu1 + other.nu1, self.ny1 + other.ny1)
    
    def __mul__(self, other):
        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Can only multiply PartitionedStateSpace objects")

        A = np.block([
            [self.A, self.B1 @ other.C1],
            [np.zeros((other.A.shape[0], self.A.shape[1])), other.A]
        ])

        B = np.block([
            [self.B1 @ other.D11, self.B2, self.B1 @ other.D12],
            [other.B1, np.zeros((other.B2.shape[0], self.B2.shape[1])), other.B2]
        ])

        C = np.block([
            [self.C1, self.D11 @ other.C1],
            [self.C2, self.D21 @ other.C1],
            [np.zeros((other.C2.shape[0], self.C2.shape[1])), other.C2]
        ])

        D = np.block([
            [self.D11 @ other.D11, self.D12, self.D11 @ other.D12],
            [self.D21 @ other.D11, self.D22, self.D21 @ other.D12],
            [other.D21, np.zeros((other.D22.shape[0], self.D22.shape[1])), other.D22]
        ])

        P = ss(A, B, C, D)
        return PartitionedStateSpace(P, other.nu1, self.ny1)

    def feedback(self, other):
        """
        Feedback connection of two PartitionedStateSpace systems.
        """
        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Feedback connection only defined for PartitionedStateSpace objects.")

        # Pre-calculate repeated inverses
        I1 = np.eye(other.D11.shape[0])
        I2 = np.eye(self.D11.shape[0])

        try:
            inv_I_plus_D211_D111 = inv(I1 + other.D11 @ self.D11)
            inv_I_plus_D111_D211 = inv(I2 + self.D11 @ other.D11)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered. Feedback connection may not be well-posed.")


        X_11 = inv_I_plus_D211_D111 @ (-other.D11 @ self.C1 - other.C1)
        X_21 = inv_I_plus_D111_D211 @ (self.C1 - self.D11 @ other.C1)
        X_12 = inv_I_plus_D211_D111 @ (I1 - other.D11 @ self.D12 - other.D12)
        X_22 = inv_I_plus_D111_D211 @ (self.D11 + self.D12 - self.D11 @ other.D12)


        A = np.block([
            [self.A + self.B1 @ X_11, self.B1 @ X_11[ : , self.C1.shape[1]:]],
            [other.B1 @ X_21[:other.C1.shape[0], :], other.A + other.B1 @ X_21]
        ])


        B = np.block([
            [self.B1 @ X_12, self.B1 @ X_12[:, I1.shape[1]: ] + self.B2],
            [other.B1 @ X_22[:, :self.D11.shape[0]], other.B1 @ X_22[:, self.D11.shape[0]:] + other.B2]
        ])

        C = np.block([
            [self.C1 + self.D11 @ X_11, self.D11 @ X_11[:, self.C1.shape[1]:]],
            [self.C2 + self.D21 @ X_11, self.D21 @ X_11[:, self.C1.shape[1]:]],
            [other.D21 @ X_21, other.C2 + other.D21 @ X_21[:, other.C1.shape[1]:]]
        ])

        D = np.block([
            [self.D11 @ X_12, self.D11 @ X_12[:, I1.shape[1]:] + self.D12],
            [self.D21 @ X_12, self.D21 @ X_12[:, I1.shape[1]:] + self.D22],
            [other.D21 @ X_22[:, :self.D11.shape[0]], other.D21 @ X_22[:, self.D11.shape[0]:] + other.D22]
        ])


        P = ss(A, B, C, D)
        return PartitionedStateSpace(P, other.nu1, self.ny1)




