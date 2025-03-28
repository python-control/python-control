import numpy as np
from scipy.linalg import block_diag, inv, solve, LinAlgError

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

        self.nstates = sys.nstates
        self.noutputs_total = sys.noutputs
        self.ninputs_total = sys.ninputs
        self.nu2 = self.ninputs_total - self.nu1 # Dimension of external input w
        self.ny2 = self.noutputs_total - self.ny1 # Dimension of external output z

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
        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Feedback connection only defined for PartitionedStateSpace objects.")

        # Pre-calculate repeated inverses
        I_self = np.eye(self.D11.shape[0])
        I_other = np.eye(other.D11.shape[0])

        X_11 = solve(I_other + other.D11 @ self.D11, np.hstack((-other.D11 @ self.C1, -other.C1)))
        X_21 = solve(I_self + self.D11 @ other.D11, np.hstack((self.C1, -self.D11 @ other.C1)))

        X_12 = solve(I_other + other.D11 @ self.D11, np.hstack((I_other, -other.D11 @ self.D12, -other.D12))) # maybe I_other
        X_22 = solve(I_self + self.D11 @ other.D11, np.hstack((self.D11, self.D12, -self.D11 @ other.D12)))

        A_new = np.vstack((self.B1 @ X_11, other.B1 @ X_21)) + block_diag(self.A, other.A)

        B_new = np.vstack((self.B1 @ X_12, other.B1 @ X_22))
        tmp = block_diag(self.B2, other.B2)
        B_new[:, -tmp.shape[1]:] += tmp

        C_new = np.vstack([
            self.D11 @ X_11,
            self.D21 @ X_11,
            other.D21 @ X_21,
        ]) + np.vstack([
            np.hstack([self.C1, np.zeros((self.C1.shape[0], other.C1.shape[1]))]),
            block_diag(self.C2, other.C2),
        ])

        D_new = np.vstack([
            self.D11 @ X_12,
            self.D21 @ X_12,
            other.D21 @ X_22,
        ])
        tmp = np.vstack([
            np.hstack([self.D12, np.zeros((self.D12.shape[0], other.D12.shape[1]))]),
            block_diag(self.D22, other.D22),
        ])
        D_new[:, -tmp.shape[1]:] += tmp

        P_new = StateSpace(A_new, B_new, C_new, D_new)

        return PartitionedStateSpace(P_new, other.nu1, self.ny1)


def vcat_pss(*systems: list[PartitionedStateSpace]) -> PartitionedStateSpace:
    
    if not all(isinstance(pss, PartitionedStateSpace) for pss in systems):
        raise TypeError("All arguments must be PartitionedStateSpace objects")

    # Not used, to be checked
    nu1 = systems[0].nu1
    ny1 = sum(space.ny1 for space in systems)

    if not (all(space.nu1 == nu1 for space in systems)):
        raise ValueError("All PartitionedStateSpace objects must have the same input dimension")
    
    A = block_diag(*[space.A for space in systems])
    B1 = np.vstack([space.B1 for space in systems])
    B2 = block_diag(*[space.B2 for space in systems])
    C1 = block_diag(*[space.C1 for space in systems])
    C2 = block_diag(*[space.C2 for space in systems])
    D11 = np.vstack([space.D11 for space in systems])
    D12 = block_diag(*[space.D12 for space in systems])
    D21 = np.vstack([space.D21 for space in systems])
    D22 = block_diag(*[space.D22 for space in systems])

    return PartitionedStateSpace.from_matrices(A, B1, B2, C1, C2, D11, D12, D21, D22)


def hcat_pss(*systems: list[PartitionedStateSpace]) -> PartitionedStateSpace:

    nu1 = sum(space.nu1 for space in systems)
    ny1 = systems[0].ny1
    if not (all(space.ny1 == ny1 for space in systems)):
        raise ValueError("All PartitionedStateSpace objects must have the same output dimension")
    
    A = block_diag(*[space.A for space in systems])
    B1 = block_diag(*[space.B1 for space in systems])
    B2 = block_diag(*[space.B2 for space in systems])
    C1 = np.hstack([space.C1 for space in systems])
    C2 = block_diag(*[space.C2 for space in systems])
    D11 = np.hstack([space.D11 for space in systems])
    D12 = np.hstack([space.D12 for space in systems])
    D21 = block_diag(*[space.D21 for space in systems])
    D22 = block_diag(*[space.D22 for space in systems])

    return PartitionedStateSpace.from_matrices(A, B1, B2, C1, C2, D11, D12, D21, D22)


