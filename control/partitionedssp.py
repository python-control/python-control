# partitionedssp.py - PartitionedStateSpace class and functions
# for Partitioned state-space systems

"""PartitionedStateSpace class
and functions for Partitioned state-space systems

This module contains the PartitionedStateSpace class and
functions for creating and manipulating partitioned state-space systems.
This class is needed to handle systems with time delays (delayLTI class).
"""

import numpy as np
from scipy.linalg import block_diag, solve

from .statesp import ss, StateSpace


class PartitionedStateSpace:
    """Partitioned State Space class.

    The PartitionedStateSpace class represents a state-space system
    partitioned into two parts: external and internal. It is used to
    handle systems with time delays.

    Parameters
    ----------
    sys : StateSpace
        The underlying state-space representation of the system.
    nu1 : int
        The number of external inputs.
    ny1 : int
        The number of external outputs.

    Attributes
    ----------
    sys : StateSpace
        The underlying state-space representation of the system.
    nu1 : int
        The number of external inputs.
    ny1 : int
        The number of external outputs.
    A : array_like
        The state matrix.
    B : array_like
        The input matrix.
    C : array_like
        The output matrix.
    D : array_like
        The direct feedthrough matrix.
    B1 : array_like
        The input matrix for external inputs.
    B2 : array_like
        The input matrix for delayed inputs.
    C1 : array_like
        The output matrix for external outputs.
    C2 : array_like
        The output matrix for delayed outputs.
    D11 : array_like
        The direct feedthrough matrix for external inputs to external outputs.
    D12 : array_like
        The direct feedthrough matrix for delayed inputs to external outputs.
    D21 : array_like
        The direct feedthrough matrix for external inputs to delayed outputs.
    D22 : array_like
        The direct feedthrough matrix for delayed inputs to delayed outputs.
    nstates : int
        The number of states.
    noutputs_total : int
        The total number of outputs.
    ninputs_total : int
        The total number of inputs.
    nu2 : int
        The number of delayed inputs.
    ny2 : int
        The number of delayed outputs.

    Methods
    -------
    from_matrices(A, B1, B2, C1, C2, D11, D12, D21, D22)
        Create a PartitionedStateSpace system from matrices.
    """

    def __init__(self, sys: StateSpace, nu1: int, ny1: int):
        """Initialize the PartitionedStateSpace object.

        Parameters
        ----------
        sys : StateSpace
            The underlying state-space representation of the system.
        nu1 : int
            The number of external inputs.
        ny1 : int
            The number of external outputs.

        Raises
        ------
        TypeError
            If the input is not a StateSpace object.
        ValueError
            If the number of external inputs or outputs is invalid.
        """
        if not isinstance(sys, StateSpace):
            raise TypeError("Input must be a StateSpace")
        if nu1 > sys.ninputs or nu1 < 0:
            raise ValueError("Invalid number of external inputs")
        if ny1 > sys.noutputs or ny1 < 0:
            raise ValueError("Invalid number of external outputs")

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

        # Dimension of external input w
        self.nu2 = self.ninputs_total - self.nu1
        # Dimension of external output z
        self.ny2 = self.noutputs_total - self.ny1

    @property
    def B1(self):
        return self.B[:, : self.nu1]

    @property
    def B2(self):
        return self.B[:, self.nu1:]

    @property
    def C1(self):
        return self.C[: self.ny1, :]

    @property
    def C2(self):
        return self.C[self.ny1:, :]

    @property
    def D11(self):
        return self.D[: self.ny1, : self.nu1]

    @property
    def D12(self):
        return self.D[: self.ny1, self.nu1:]

    @property
    def D21(self):
        return self.D[self.ny1:, : self.nu1]

    @property
    def D22(self):
        return self.D[self.ny1:, self.nu1:]

    @classmethod
    def from_matrices(cls, A, B1, B2, C1, C2, D11, D12, D21, D22):
        """Create a PartitionedStateSpace system from matrices.

        Parameters
        ----------
        A : array_like
            The state matrix.
        B1 : array_like
            The input matrix for external inputs.
        B2 : array_like
            The input matrix for delayed inputs.
        C1 : array_like
            The output matrix for external outputs.
        C2 : array_like
            The output matrix for delayed outputs.
        D11 : array_like
            The direct feedthrough matrix for external inputs
            to external outputs.
        D12 : array_like
            The direct feedthrough matrix for delayed inputs
            to external outputs.
        D21 : array_like
            The direct feedthrough matrix for external inputs
            to delayed outputs.
        D22 : array_like
            The direct feedthrough matrix for delayed inputs
            to delayed outputs (should be zeros for
            delay LTI).

        Returns
        -------
        PartitionedStateSpace
            The PartitionedStateSpace system.

        Raises
        ------
        ValueError
            If the matrices have incompatible shapes.
        """

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
        """Add two PartitionedStateSpace systems.

        Parameters
        ----------
        other : PartitionedStateSpace
            The other system to add.

        Returns
        -------
        PartitionedStateSpace
            The resulting PartitionedStateSpace system.

        Raises
        ------
        TypeError
            If the operand type is not supported.
        """

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
        return PartitionedStateSpace(
            P, self.nu1 + other.nu1, self.ny1 + other.ny1
        )

    def __mul__(self, other):
        """Multiply two PartitionedStateSpace systems.

        Parameters
        ----------
        other : PartitionedStateSpace
            The other system to multiply with.

        Returns
        -------
        PartitionedStateSpace
            The resulting PartitionedStateSpace system.

        Raises
        ------
        TypeError
            If the operand type is not supported.
        """

        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Can only multiply PartitionedStateSpace objects")

        A = np.block(
            [
                [self.A, self.B1 @ other.C1],
                [np.zeros((other.A.shape[0], self.A.shape[1])), other.A],
            ]
        )

        B = np.block(
            [
                [
                    self.B1 @ other.D11,
                    self.B2, self.B1 @ other.D12
                ],
                [
                    other.B1,
                    np.zeros((other.B2.shape[0], self.B2.shape[1])),
                    other.B2
                ]
            ]
        )

        C = np.block(
            [
                [
                    self.C1,
                    self.D11 @ other.C1,
                ],
                [
                    self.C2,
                    self.D21 @ other.C1
                ],
                [
                    np.zeros((other.C2.shape[0], self.C2.shape[1])),
                    other.C2
                ]
            ]
        )

        D = np.block(
            [
                [self.D11 @ other.D11, self.D12, self.D11 @ other.D12],
                [self.D21 @ other.D11, self.D22, self.D21 @ other.D12],
                [
                    other.D21,
                    np.zeros((other.D22.shape[0], self.D22.shape[1])),
                    other.D22,
                ],
            ]
        )

        P = ss(A, B, C, D)
        return PartitionedStateSpace(P, other.nu1, self.ny1)

    def __eq__(self, other):
        return (
            np.allclose(self.A, other.A)
            and np.allclose(self.B, other.B)
            and np.allclose(self.C, other.C)
            and np.allclose(self.D, other.D)
            and self.nu1 == other.nu1
            and self.ny1 == other.ny1
        )

    def feedback(self, other):
        """Feedback interconnection for PartitionedStateSpace.

        Parameters
        ----------
        other : PartitionedStateSpace
            The system in the feedback path.

        Returns
        -------
        PartitionedStateSpace
            The resulting PartitionedStateSpace system.

        Raises
        ------
        TypeError
            If the operand type is not supported.
        """

        if not isinstance(other, PartitionedStateSpace):
            raise TypeError("Feedback connection only defined\
                            for PartitionedStateSpace objects.")

        # Pre-calculate repeated inverses
        I_self = np.eye(self.D11.shape[0])
        I_other = np.eye(other.D11.shape[0])

        X_11 = solve(
            I_other + other.D11 @ self.D11,
            np.hstack((-other.D11 @ self.C1, -other.C1))
        )
        X_21 = solve(
            I_self + self.D11 @ other.D11,
            np.hstack((self.C1, -self.D11 @ other.C1))
        )

        X_12 = solve(
            I_other + other.D11 @ self.D11,
            np.hstack((I_other, -other.D11 @ self.D12, -other.D12)),
        )  # maybe I_other
        X_22 = solve(
            I_self + self.D11 @ other.D11,
            np.hstack((self.D11, self.D12, -self.D11 @ other.D12)),
        )

        A_new = np.vstack((self.B1 @ X_11, other.B1 @ X_21)) + \
            block_diag(self.A, other.A)

        B_new = np.vstack((self.B1 @ X_12, other.B1 @ X_22))
        tmp = block_diag(self.B2, other.B2)
        B_new[:, -tmp.shape[1]:] += tmp

        C_new = np.vstack([
                self.D11 @ X_11,
                self.D21 @ X_11,
                other.D21 @ X_21,
        ]) + np.vstack([
                np.hstack([
                    self.C1,
                    np.zeros((self.C1.shape[0], other.C1.shape[1]))
                ]),
                block_diag(self.C2, other.C2),
        ])

        D_new = np.vstack([
                self.D11 @ X_12,
                self.D21 @ X_12,
                other.D21 @ X_22,
        ])
        tmp = np.vstack([
                np.hstack([
                    self.D12,
                    np.zeros((self.D12.shape[0], other.D12.shape[1]))
                ]),
                block_diag(self.D22, other.D22),
        ])
        D_new[:, -tmp.shape[1]:] += tmp

        P_new = StateSpace(A_new, B_new, C_new, D_new)

        return PartitionedStateSpace(P_new, other.nu1, self.ny1)

    def __str__(self):
        s = "PartitionedStateSpace\n"
        s += "A = \n"
        s += str(self.A)
        s += "\nB = \n"
        s += str(self.B)
        s += "\nC = \n"
        s += str(self.C)
        s += "\nD = \n"
        s += str(self.D)
        s += "\n"
        return s


def vcat_pss(*systems: list[PartitionedStateSpace]) -> PartitionedStateSpace:
    """Vertically concatenate a list of PartitionedStateSpace systems.

    Parameters
    ----------
    *systems : list of PartitionedStateSpace
        The systems to be concatenated.

    Returns
    -------
    PartitionedStateSpace
        The resulting PartitionedStateSpace system.

    Raises
    ------
    TypeError
        If any of the inputs are not PartitionedStateSpace systems.
    ValueError
        If the systems do not have the same number of inputs.
    """

    if not all(isinstance(pss, PartitionedStateSpace) for pss in systems):
        raise TypeError("All arguments must be PartitionedStateSpace objects")

    nu1 = systems[0].nu1

    if not (all(space.nu1 == nu1 for space in systems)):
        raise ValueError("All PartitionedStateSpace objects\
                          must have the same input dimension")

    A = block_diag(*[space.A for space in systems])
    B1 = np.vstack([space.B1 for space in systems])
    B2 = block_diag(*[space.B2 for space in systems])
    C1 = block_diag(*[space.C1 for space in systems])
    C2 = block_diag(*[space.C2 for space in systems])
    D11 = np.vstack([space.D11 for space in systems])
    D12 = block_diag(*[space.D12 for space in systems])
    D21 = np.vstack([space.D21 for space in systems])
    D22 = block_diag(*[space.D22 for space in systems])

    return PartitionedStateSpace.from_matrices(
        A, B1, B2, C1, C2, D11, D12, D21, D22
    )


def hcat_pss(*systems: list[PartitionedStateSpace]) -> PartitionedStateSpace:
    """Horizontally concatenate a list of PartitionedStateSpace systems.

    Parameters
    ----------
    *systems : list of PartitionedStateSpace
        The systems to be concatenated.

    Returns
    -------
    PartitionedStateSpace
        The resulting PartitionedStateSpace system.

    Raises
    ------
    TypeError
        If any of the inputs are not PartitionedStateSpace systems.
    ValueError
        If the systems do not have the same number of outputs.
    """
    if not all(isinstance(pss, PartitionedStateSpace) for pss in systems):
        raise TypeError("All arguments must be PartitionedStateSpace objects")

    ny1 = systems[0].ny1
    if not (all(space.ny1 == ny1 for space in systems)):
        raise ValueError("All PartitionedStateSpace objects\
                          must have the same output dimension")

    A = block_diag(*[space.A for space in systems])
    B1 = block_diag(*[space.B1 for space in systems])
    B2 = block_diag(*[space.B2 for space in systems])
    C1 = np.hstack([space.C1 for space in systems])
    C2 = block_diag(*[space.C2 for space in systems])
    D11 = np.hstack([space.D11 for space in systems])
    D12 = np.hstack([space.D12 for space in systems])
    D21 = block_diag(*[space.D21 for space in systems])
    D22 = block_diag(*[space.D22 for space in systems])

    return PartitionedStateSpace.from_matrices(
        A, B1, B2, C1, C2, D11, D12, D21, D22
    )
