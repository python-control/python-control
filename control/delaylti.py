import numpy as np
from .lti import LTI
from .partitionedssp import PartitionedStateSpace
from .statesp import ss, StateSpace, tf2ss
from .xferfcn import TransferFunction
from .iosys import _process_iosys_keywords
from scipy.linalg import solve, LinAlgError, inv, eigvals



class DelayLTI(LTI):
    """Delay Linear Time Invariant (DelayLTI) class.

    The DelayLTI class is a subclass of the LTI class that represents a
    linear time-invariant (LTI) system with time delays. It is designed to
    handle systems where the output depends not only on the current input
    but also on past inputs.

    Parameters
    ----------
    P : PartitionedStateSpace
        The underlying partitioned state-space representation of the system.
    tau : array_like, optional
        An array of time delays associated with the system.
    **kwargs : keyword arguments
        Additional keyword arguments for the LTI system.

    Attributes
    ----------
    P : PartitionedStateSpace
        The underlying partitioned state-space representation of the system.
    tau : array_like
        An array of time delays associated with the system.
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
    ninputs : int
        The number of external inputs.
    noutputs : int
        The number of external outputs.
    nstates : int
        The number of states.

    Methods
    -------
    from_ss(sys, tau)
        Create a DelayLTI system from a StateSpace system.
    from_tf(sys, tau)
        Create a DelayLTI system from a TransferFunction system.
    size()
        Return the number of outputs and inputs.
    """

    def __init__(self, P: PartitionedStateSpace, tau = None, **kwargs):
        """Initialize the DelayLTI object.

        Parameters
        ----------
        P : PartitionedStateSpace
            The underlying partitioned state-space representation of the system.
        tau : array_like, optional
            An array of time delays associated with the system.
        **kwargs : keyword arguments
            Additional keyword arguments for the LTI system.

        """
        if not isinstance(P, PartitionedStateSpace):
            raise TypeError("Input must be a PartitionedStateSpace")
        
        self.P = P
        self.tau = np.array([]) if tau is None else np.array(tau)
        
        static = (self.P.A.size == 0)

        self.nu = self.P.sys.ninputs - len(self.tau)
        self.ny = self.P.sys.noutputs - len(self.tau)

        super().__init__(self.nu, self.ny, self.P.sys.nstates)

    @classmethod
    def from_ss(cls, sys: StateSpace, tau: np.ndarray = None):
        """Create a DelayLTI system from a StateSpace system.

        Parameters
        ----------
        sys : StateSpace
            The underlying state-space representation of the system.
        tau : array_like, optional
            An array of time delays associated with the system.

        Returns
        -------
        DelayLTI
            The DelayLTI system.

        """
        if not isinstance(sys, StateSpace):
            raise TypeError("Input must be a StateSpace")
        
        tau = np.array([]) if tau is None else np.array(tau)
        
        nu = sys.D.shape[1] - len(tau)
        ny = sys.D.shape[0] - len(tau)

        if nu < 0 or ny < 0:
            raise ValueError("tau is too long")

        psys = PartitionedStateSpace(sys, nu, ny)
        return cls(psys, tau)
    
    @classmethod
    def from_tf(cls, sys: TransferFunction, tau: np.ndarray = None):
        """Create a DelayLTI system from a TransferFunction system.

        Parameters
        ----------
        sys : TransferFunction
            The underlying transfer function representation of the system.
        tau : array_like, optional
            An array of time delays associated with the system.

        Returns
        -------
        DelayLTI
            The DelayLTI system.

        """
        if not isinstance(sys, TransferFunction):
            raise TypeError("Input must be a TransferFunction")
        return DelayLTI.from_ss(tf2ss(sys), tau)

    def size(self):
        """Return the number of outputs and inputs."""
        return (self.noutputs, self.ninputs)
    
    def poles(self):
        """Compute the poles of a delay lti system."""

        return eigvals(self.P.A).astype(complex) if self.nstates \
            else np.array([])

    def zeros(self):
        """Compute the zeros of a delay lti system."""

        if not self.nstates:
            return np.array([])

        # Use AB08ND from Slycot if it's available, otherwise use
        # scipy.lingalg.eigvals().
        try:
            from slycot import ab08nd

            out = ab08nd(self.P.A.shape[0], self.P.B.shape[1], self.P.C.shape[0],
                         self.P.A, self.P.B, self.P.C, self.P.D)
            nu = out[0]
            if nu == 0:
                return np.array([])
            else:
                # Use SciPy generalized eigenvalue function
                return eigvals(out[8][0:nu, 0:nu],
                                         out[9][0:nu, 0:nu]).astype(complex)

        except ImportError:  # Slycot unavailable. Fall back to SciPy.
            if self.P.C.shape[0] != self.P.D.shape[1]:
                raise NotImplementedError(
                    "StateSpace.zero only supports systems with the same "
                    "number of inputs as outputs.")

            # This implements the QZ algorithm for finding transmission zeros
            # from
            # https://dspace.mit.edu/bitstream/handle/1721.1/841/P-0802-06587335.pdf.
            # The QZ algorithm solves the generalized eigenvalue problem: given
            # `L = [A, B; C, D]` and `M = [I_nxn 0]`, find all finite lambda
            # for which there exist nontrivial solutions of the equation
            # `Lz - lamba Mz`.
            #
            # The generalized eigenvalue problem is only solvable if its
            # arguments are square matrices.
            L = np.concatenate((np.concatenate((self.P.A, self.P.B), axis=1),
                             np.concatenate((self.P.C, self.P.D), axis=1)), axis=0)
            M = np.pad(np.eye(self.P.A.shape[0]), ((0, self.P.C.shape[0]),
                                           (0, self.P.B.shape[1])), "constant")
            return np.array([x for x in eigvals(L, M,
                                                          overwrite_a=True)
                             if not np.isinf(x)], dtype=complex)
        
    def _isstatic(self):
        """Check if the system is static."""
        return self.nstates == 0

    def __mul__(self, other):
        """Multiply two DelayLTI systems or a DelayLTI system with a scalar.

        Parameters
        ----------
        other : DelayLTI, scalar, TransferFunction, StateSpace
            The other system or scalar to multiply with.

        Returns
        -------
        DelayLTI
            The resulting DelayLTI system.

        Raises
        ------
        TypeError
            If the operand type is not supported.
        """

        if isinstance(other, (int, float, complex)):
            new_C = np.block([[self.P.C1 * other], [self.P.C2]])
            new_D = np.block([[self.P.D11 * other, self.P.D12 * other], [self.P.D21, self.P.D22]])
            new_P = PartitionedStateSpace(ss(self.P.A, self.P.B, new_C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTI(new_P, self.tau)

        elif isinstance(other, DelayLTI):
            psys_new = self.P * other.P
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTI(psys_new, tau_new)
        
        elif isinstance(other, TransferFunction):
            dlti = tf2dlti(other)
            return self * dlti
        
        elif isinstance(other, StateSpace):
            return self * DelayLTI.from_ss(other)

        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))
        
    def __rmul__(self, other):
        
        if isinstance(other, (int, float, complex)):
            return self * other
        
        elif isinstance(other, TransferFunction):
            dlti = tf2dlti(other)
            return dlti * self
        
        elif isinstance(other, StateSpace):
            return DelayLTI.from_ss(other) * self

        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(other), type(self)))

    def __add__(self, other):
        """Add two DelayLTI systems or a DelayLTI system with a scalar.

        Parameters
        ----------
        other : DelayLTI, scalar, TransferFunction, StateSpace
            The other system or scalar to add.

        Returns
        -------
        DelayLTI
            The resulting DelayLTI system.

        Raises
        ------
        TypeError
            If the operand type is not supported.
        """

        if isinstance(other, (int, float, complex)):
            new_D = self.P.sys.D.copy()
            new_D[:self.ny, :self.nu] += other
            pnew = PartitionedStateSpace(ss(self.P.A, self.P.B, self.P.C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTI(pnew, self.tau)
        elif isinstance(other, DelayLTI):
            psys_new = self.P + other.P
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTI(psys_new, tau_new)
        else:
            sys = _convert_to_delay_lti(other)
            return self + sys
            

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        return -self + other

    def __eq__(self, other):
        if not isinstance(other, DelayLTI):
            raise TypeError(f"{other} is not a DelayLTI object, is {type(other)}")
        return (self.P == other.P) and (self.tau == other.tau)
    
    def feedback(self, other=1, sign=-1):
        """Standard or LFT feedback interconnection for DelayLTI.

        If `other` is a static gain (scalar, matrix, or static LTI system),
        computes the standard feedback loop: u = r + sign*other*y, y = self*u.
        The resulting system maps the external input `r` and the internal
        delay input `w` to the external output `y` and the internal delay
        output `z`.

        If `other` is also a `DelayLTI`, computes the LFT feedback
        interconnection by calling `feedback` on the underlying
        `PartitionedStateSpace` objects and concatenating the delay vectors.

        Parameters
        ----------
        other : scalar, array, LTI system, or DelayLTI
            The system or gain in the feedback path.
        sign : int, optional {-1, 1}
            Sign of the feedback. Default is -1 (negative feedback).

        Returns
        -------
        DelayLTI
            The closed-loop system.
        """

        if isinstance(other, DelayLTI):
            psys_new = self.P.feedback(other.P)
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTI(psys_new, tau_new)
        
        elif isinstance(other, (StateSpace, TransferFunction)):
            other_delay_lti = _convert_to_delay_lti(other)
            return self.feedback(other_delay_lti)
        
        else:
            # Convert feedback 'other' to a static gain matrix K
            if isinstance(other, (int, float, complex, np.number)):
                if not self.issiso():
                     raise ValueError("Scalar feedback gain requires SISO system G.")
                K = np.array([[other]], dtype=float)
            elif isinstance(other, np.ndarray):
                K = np.asarray(other, dtype=float)
                if K.ndim == 0: K = K.reshape(1,1)
                elif K.ndim == 1:
                     if self.nu != 1: raise ValueError("1D array feedback requires SISO system G.")
                     K = K.reshape(self.ninputs, 1)
                elif K.ndim != 2: raise ValueError("Feedback gain must be scalar, 1D, or 2D array.")
            else:
                 raise TypeError(f"Unsupported type for static feedback: {type(other)}")

            # Check dimensions of K
            if K.shape != (self.nu, self.ny):
                raise ValueError(f"Feedback gain K has incompatible shape. Expected ({self.nu}, {self.ny}), got {K.shape}.")

            # Get matrices from self's underlying PartitionedStateSpace
            P_g = self.P
            A_g, B1_g, B2_g = P_g.A, P_g.B1, P_g.B2
            C1_g, C2_g = P_g.C1, P_g.C2
            D11_g, D12_g = P_g.D11, P_g.D12
            D21_g, D22_g = P_g.D21, P_g.D22
            taus = self.tau
            n_states = self.nstates
            n_w = B2_g.shape[1]     # Delay input dimension
            n_z = C2_g.shape[0]     # Delay output dimension
            n_r = self.nu           # Reference input dimension

            # Promote types, handle empty states
            T = np.promote_types(A_g.dtype if A_g.size > 0 else float, K.dtype)
            if n_states == 0: A_g = np.zeros((0,0), dtype=T)

            # Calculate closed-loop matrices for map [r, w] -> [y, z]
            F = np.eye(self.nu, dtype=T) - sign * K @ D11_g
            try:
                invF_signK = solve(F, sign * K)
                invF = solve(F, np.eye(self.nu, dtype=T))
            except LinAlgError:
                raise ValueError("Algebraic loop; I - sign*K*D11 is singular.")

            A_new = A_g + B1_g @ invF_signK @ C1_g
            B1_new = B1_g @ invF
            B2_new = B2_g + B1_g @ invF_signK @ D12_g
            C1_new = C1_g + D11_g @ invF_signK @ C1_g
            C2_new = C2_g + D21_g @ invF_signK @ C1_g
            D11_new = D11_g @ invF
            D12_new = D12_g + D11_g @ invF_signK @ D12_g
            D21_new = D21_g @ invF
            D22_new = D22_g + D21_g @ invF_signK @ D12_g

            B_new = np.hstack([B1_new, B2_new]) if B1_new.size > 0 or B2_new.size > 0 else np.zeros((n_states, n_r + n_w), dtype=T)
            C_new = np.vstack([C1_new, C2_new]) if C1_new.size > 0 or C2_new.size > 0 else np.zeros((self.ny + n_z, n_states), dtype=T)
            D_new = np.block([[D11_new, D12_new], [D21_new, D22_new]]) if D11_new.size>0 or D12_new.size>0 or D21_new.size>0 or D22_new.size>0 else np.zeros((self.ny + n_z, n_r + n_w), dtype=T)

            clsys_ss = StateSpace(A_new, B_new, C_new, D_new, self.dt)
            clsys_part = PartitionedStateSpace(clsys_ss, nu1=n_r, ny1=self.ny)
            return DelayLTI(clsys_part, taus)
    
    def issiso(self):
        """Check if the system is single-input, single-output."""
        # Based on EXTERNAL dimensions
        return self.nu == 1 and self.ny == 1
    
    def __call__(self, x, squeeze=False, warn_infinite=True):
        """Evaluate the frequency response of the system.

        Parameters
        ----------
        x : array_like
            Complex frequencies at which to evaluate the frequency response.
        squeeze : bool, optional
            If squeeze=True, access to the output response will remove
            single-dimensional entries from the shape of the inputs,
            outputs, and states even if the system is not SISO. If
            squeeze=False, keep the input as a 2D or 3D array (indexed
            by the input (if multi-input), trace (if single input) and
            time) and the output and states as a 3D array (indexed by the
            output/state, trace, and time) even if the system is SISO.
        warn_infinite : bool, optional
            If True, issue a warning if an infinite value is found in the
            frequency response.

        Returns
        -------
        out : array_like
            Frequency response of the system.
        """
        x_arr = np.atleast_1d(x).astype(complex, copy=False)

        if len(x_arr.shape) > 1:
            raise ValueError("input list must be 1D")
        
        out = np.empty((self.ny, self.nu, len(x_arr)), dtype=complex)

        sys_call = self.P.sys(x_arr, squeeze=squeeze, warn_infinite=warn_infinite)
        for i, xi in enumerate(x_arr):
            P11_fr = sys_call[:self.ny, :self.nu, i]
            P12_fr = sys_call[:self.ny, self.nu:, i]
            P21_fr = sys_call[self.ny:, :self.nu, i]
            P22_fr = sys_call[self.ny:, self.nu:, i]
            delay_term_inv = np.exp(xi * self.tau)
            delay_term_fr = np.diag(delay_term_inv)
            out[:,:,i] = P11_fr + P12_fr @  inv(delay_term_fr - P22_fr) @ P21_fr
        return out

    def __str__(self):
        # To be improved
        s = f"DelayLTI with {self.noutputs} outputs, {self.ninputs} inputs, " \
            f"{self.nstates} states, and {len(self.tau)} delays.\n"
        s += f"Delays: {self.tau}\n"
        s += "Underlying PartitionedStateSpace P:\n" + str(self.P)
        s += "\n"
        return s

    def __repr__(self):
         return (f"{type(self).__name__}("
                 f"P={self.P.__repr__()}, tau={self.tau.__repr__()})")


def delay(tau):
    """
    Create a pure delay system.

    Parameters
    ----------
    tau : float, list, or NumPy array
        The time delay(s) for the system. If a list or NumPy array is
        provided, each element represents a separate delay.

    Returns
    -------
    DelayLTI
        A DelayLTI system representing the pure delay.

    Raises
    ------
    TypeError
        If tau is not a number, list, or NumPy array.

    """

    if isinstance(tau, (int, float)):
        tau_arr = [float(tau)]
        num_delays = 1
    elif isinstance(tau, (list, np.ndarray)):
        tau_arr = [float(t) for t in tau] 
        num_delays = len(tau_arr)
    else:
        raise TypeError("tau must be a number, list, or NumPy array")
    
    D = np.array([
        [0, 1],
        [1, 0]
    ])

    ny, nu = D.shape[0], D.shape[1]

    A = np.zeros((0,0))
    B = np.zeros((0, nu))
    C = np.zeros((ny, 0))

    P = PartitionedStateSpace(ss(A, B, C, D), 1, 1)
    return DelayLTI(P, tau_arr)


def exp(G):
    """
    Create delay in the form of exp(-τ*s) where s=tf("s")

    Parameters
    ----------
    G : TransferFunction
        The transfer function representing the delay.

    Returns
    -------
    DelayLTI
        A DelayLTI system representing the pure delay.

    Raises
    ------
    ValueError
        If the input is not of the form -τ*s, τ>0.
    """
    num = G.num[0][0]
    den = G.den[0][0]

    if not (len(den) == 1 and len(num) == 2 and num[0] < 0 and num[1] == 0):
        raise ValueError("Input must be of the form -τ*s, τ>0.")

    return delay(-num[0] / den[0])


def tf2dlti(tf: TransferFunction):
    """
    Convert a TransferFunction to a DelayLTI
    """
    if not isinstance(tf, TransferFunction):
        raise TypeError("Input must be a TransferFunction")
    
    ss_tf = tf2ss(tf)
    return DelayLTI.from_ss(ss_tf)


def ss2dlti(ss: StateSpace):
    """
    Convert a StateSpace to a DelayLTI
    """
    return DelayLTI.from_ss(ss)


def _convert_to_delay_lti(sys):
    """Convert a system to a DelayLTI if necessary."""
    if isinstance(sys, DelayLTI):
        return sys
    elif isinstance(sys, StateSpace):
        return DelayLTI.from_ss(sys)
    elif isinstance(sys, TransferFunction):
        return tf2dlti(sys)
    else:
        raise TypeError("Unsupported system type for DelayLTI conversion: {}".format(type(sys)))
    

def vcat(*systems: list[DelayLTI]) -> DelayLTI:
    """Vertically concatenate a list of DelayLTI systems.

    Parameters
    ----------
    *systems : list of DelayLTI
        The systems to be concatenated.

    Returns
    -------
    DelayLTI
        The resulting DelayLTI system.

    Raises
    ------
    TypeError
        If any of the inputs are not DelayLTI systems.
    """

    from .partitionedssp import vcat_pss

    if not all(isinstance(sys, DelayLTI) for sys in systems):
        raise TypeError("All inputs must be DelayLTIs")

    part_ssp = [sys.P for sys in systems]
    P = vcat_pss(*part_ssp)
    tau = np.concatenate([sys.tau for sys in systems])
    return DelayLTI(P, tau)
    

def hcat(*systems: list[DelayLTI]) -> DelayLTI:
    """Horizontally concatenate a list of DelayLTI systems.

    Parameters
    ----------
    *systems : list of DelayLTI
        The systems to be concatenated.

    Returns
    -------
    DelayLTI
        The resulting DelayLTI system.

    Raises
    ------
    TypeError
        If any of the inputs are not DelayLTI systems.
    """

    from .partitionedssp import hcat_pss

    if not(all(isinstance(sys, DelayLTI) for sys in systems)):
        raise TypeError("All inputs must be DelayLTIs")
    
    part_ssp = [sys.P for sys in systems]
    P = hcat_pss(*part_ssp)
    tau = np.concatenate([sys.tau for sys in systems])
    return DelayLTI(P, tau)
    

def mimo_delay(array: np.ndarray[DelayLTI]):
    """Create a MIMO delay system from an array of DelayLTI systems.

    Parameters
    ----------
    array : np.ndarray of DelayLTI
        An array of DelayLTI systems.

    Returns
    -------
    DelayLTI
        The resulting DelayLTI system.

    """

    if not all(isinstance(item, DelayLTI) for row in array for item in row):
        raise TypeError("All elements in the array must be DelayLTI systems")

    rows = [hcat(*row) for row in array]
    return vcat(*rows)