import numpy as np
from .lti import LTI
from .partitionedssp import PartitionedStateSpace
from .statesp import ss, StateSpace, tf2ss
from .xferfcn import TransferFunction
from .iosys import _process_iosys_keywords
from scipy.linalg import solve, LinAlgError, inv



class DelayLTISystem(LTI):
    
    def __init__(self, P: PartitionedStateSpace, tau: np.ndarray = [], **kwargs):

        self.P = P
        self.tau = tau

        self.A = P.A
        self.B = P.B
        self.C = P.C
        self.D = P.D

        self.B1 = P.B1
        self.B2 = P.B2
        self.C1 = P.C1
        self.C2 = P.C2
        self.D11 = P.D11
        self.D12 = P.D12
        self.D21 = P.D21
        self.D22 = P.D22

        static = (self.A.size == 0)

        self.ninputs = self.P.sys.ninputs - len(self.tau)
        self.noutputs = self.P.sys.noutputs - len(self.tau)
        self.nstates = self.P.sys.nstates

        defaults = {'inputs': self.B1.shape[1], 'outputs': self.C1.shape[0],
             'states': self.A.shape[0]}
        name, inputs, outputs, states, dt = _process_iosys_keywords(
            kwargs, defaults, static=static)

        super().__init__(inputs, outputs, states, name)

    @classmethod
    def from_ss(cls, sys: StateSpace, tau: np.ndarray = []):
        nu = sys.D.shape[1] - len(tau)
        ny = sys.D.shape[0] - len(tau)

        if nu < 0 or ny < 0:
            raise ValueError("tau is too long")

        psys = PartitionedStateSpace(sys, nu, ny)
        return cls(psys, tau)
    
    @classmethod
    def from_tf(cls, sys: TransferFunction, tau: np.ndarray = []):
        return DelayLTISystem.from_ss(tf2ss(sys), tau)

    def size(self):
        return (self.noutputs, self.ninputs)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            new_C = np.block([[self.P.C1 * other], [self.P.C2]])
            new_D = np.block([[self.P.D11 * other, self.P.D12 * other], [self.P.D21, self.P.D22]])
            new_P = PartitionedStateSpace(ss(self.P.A, self.P.B, new_C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTISystem(new_P, self.tau)

        elif isinstance(other, DelayLTISystem):
            psys_new = self.P * other.P
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTISystem(psys_new, tau_new)
        
        elif isinstance(other, TransferFunction):
            dlti = tf2dlti(other)
            return self * dlti
        
        elif isinstance(other, StateSpace):
            return self * DelayLTISystem.from_ss(other)

        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self * other
        
        elif isinstance(other, TransferFunction):
            dlti = tf2dlti(other)
            return dlti * self
        
        elif isinstance(other, StateSpace):
            return DelayLTISystem.from_ss(other) * self

        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(other), type(self)))

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            new_D = self.P.sys.D.copy()
            new_D[:self.noutputs, :self.ninputs] += other
            pnew = PartitionedStateSpace(ss(self.A, self.B, self.C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTISystem(pnew, self.tau)
        elif isinstance(other, DelayLTISystem):
            psys_new = self.P + other.P
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTISystem(psys_new, tau_new)
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
        if not isinstance(other, DelayLTISystem):
            return False
        return (np.allclose(self.A, other.A) and
                np.allclose(self.B, other.B) and
                np.allclose(self.C, other.C) and
                np.allclose(self.D, other.D) and
                np.allclose(self.tau, other.tau))
    
    def feedback(self, other=1, sign=-1):
        """Standard or LFT feedback interconnection for DelayLTISystem.

        If `other` is a static gain (scalar, matrix, or static LTI system),
        computes the standard feedback loop: u = r + sign*other*y, y = self*u.
        The resulting system maps the external input `r` and the internal
        delay input `w` to the external output `y` and the internal delay
        output `z`.

        If `other` is also a `DelayLTISystem`, computes the LFT feedback
        interconnection by calling `feedback` on the underlying
        `PartitionedStateSpace` objects and concatenating the delay vectors.

        Parameters
        ----------
        other : scalar, array, LTI system, or DelayLTISystem
            The system or gain in the feedback path.
        sign : int, optional {-1, 1}
            Sign of the feedback. Default is -1 (negative feedback).

        Returns
        -------
        DelayLTISystem
            The closed-loop system.
        """

        if isinstance(other, DelayLTISystem):
            psys_new = self.P.feedback(other.P)
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTISystem(psys_new, tau_new)
        
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
                     if self.noutputs != 1: raise ValueError("1D array feedback requires SISO system G.")
                     K = K.reshape(self.ninputs, 1)
                elif K.ndim != 2: raise ValueError("Feedback gain must be scalar, 1D, or 2D array.")
            else:
                 raise TypeError(f"Unsupported type for static feedback: {type(other)}")

            # Check dimensions of K
            if K.shape != (self.ninputs, self.noutputs):
                raise ValueError(f"Feedback gain K has incompatible shape. Expected ({self.ninputs}, {self.noutputs}), got {K.shape}.")

            # Get matrices from self's underlying PartitionedStateSpace
            P_g = self.P
            A_g, B1_g, B2_g = P_g.A, P_g.B1, P_g.B2
            C1_g, C2_g = P_g.C1, P_g.C2
            D11_g, D12_g = P_g.D11, P_g.D12
            D21_g, D22_g = P_g.D21, P_g.D22
            taus = self.tau
            n_states = self.nstates
            n_u = self.ninputs      # G's external input dimension
            n_y = self.noutputs     # G's external output dimension
            n_w = B2_g.shape[1]     # Delay input dimension
            n_z = C2_g.shape[0]     # Delay output dimension
            n_r = n_u               # Reference input dimension

            # Promote types, handle empty states
            T = np.promote_types(A_g.dtype if A_g.size > 0 else float, K.dtype)
            if n_states == 0: A_g = np.zeros((0,0), dtype=T)

            # Calculate closed-loop matrices for map [r, w] -> [y, z]
            F = np.eye(n_u, dtype=T) - sign * K @ D11_g
            try:
                invF_signK = solve(F, sign * K)
                invF = solve(F, np.eye(n_u, dtype=T))
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
            C_new = np.vstack([C1_new, C2_new]) if C1_new.size > 0 or C2_new.size > 0 else np.zeros((n_y + n_z, n_states), dtype=T)
            D_new = np.block([[D11_new, D12_new], [D21_new, D22_new]]) if D11_new.size>0 or D12_new.size>0 or D21_new.size>0 or D22_new.size>0 else np.zeros((n_y + n_z, n_r + n_w), dtype=T)

            # Create the new StateSpace system
            clsys_ss = StateSpace(A_new, B_new, C_new, D_new, self.dt)

            # Partition it correctly: inputs [r, w], outputs [y, z]
            clsys_part = PartitionedStateSpace(clsys_ss, nu1=n_r, ny1=n_y)

            # Determine result types and construct the new DelayLTISystem
            # Need to promote delay type S with potential default float
            _, S = _promote_delay_system_types(self, self)
            return DelayLTISystem(clsys_part, taus)
        
    def frequency_response(self, omega=None, squeeze=None):
        from .frdata import FrequencyResponseData

        if omega is None:
            # Use default frequency range
            from .freqplot import _default_frequency_range
            omega = _default_frequency_range(self)

        omega = np.sort(np.array(omega, ndmin=1))

        ny = self.noutputs
        nu = self.ninputs
        response = np.empty((ny, nu, len(omega)), dtype=complex)

        P_fr = self.P.sys(1j * omega)
        
        for i,w in enumerate(omega):
            P11_fr = P_fr[:ny, :nu, i]
            P12_fr = P_fr[:ny, nu:, i]
            P21_fr = P_fr[ny:, :nu, i]
            P22_fr = P_fr[ny:, nu:, i]
            delay_term_inv = np.exp(1j * w * self.tau)
            delay_term_fr = np.diag(delay_term_inv)
            response[:,:,i] = P11_fr + P12_fr @  inv(delay_term_fr - P22_fr) @ P21_fr

        return FrequencyResponseData(
            response, omega, return_magphase=True, squeeze=squeeze,
            dt=self.dt, sysname=self.name, inputs=self.input_labels,
            outputs=self.output_labels, plot_type='bode')
    
    def issiso(self):
        """Check if the system is single-input, single-output."""
        # Based on EXTERNAL dimensions
        return self.ninputs == 1 and self.noutputs == 1

    def __str__(self):
        # ... (existing string representation) ...
        s = f"DelayLTISystem with {self.noutputs} outputs, {self.ninputs} inputs, " \
            f"{self.nstates} states, and {len(self.tau)} delays.\n"
        s += f"Delays: {self.tau}\n"
        s += "Underlying PartitionedStateSpace P:\n" + str(self.P)
        return s

    def __repr__(self):
         return (f"{type(self).__name__}("
                 f"P={self.P.__repr__()}, tau={self.tau.__repr__()})")


def delay(tau):
    """
    Pure delay
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
    return DelayLTISystem(P, tau_arr)


def exp(G):
    """
    create delay in the form of exp(-τ*s) where s=tf("s")
    """
    num = G.num[0][0]
    den = G.den[0][0]

    if not (len(den) == 1 and len(num) == 2 and num[0] < 0 and num[1] == 0):
        raise ValueError("Input must be of the form -τ*s, τ>0.")

    return delay(-num[0] / den[0])


def tf2dlti(tf: TransferFunction):
    """
    Convert a TransferFunction to a DelayLTISystem
    """
    if not isinstance(tf, TransferFunction):
        raise TypeError("Input must be a TransferFunction")
    
    ss_tf = tf2ss(tf)
    return DelayLTISystem.from_ss(ss_tf)
    

def _promote_delay_system_types(sys1, sys2):
    """Determine the numeric and delay types for combined systems."""
    # Promote numeric types based on underlying StateSpace
    T = np.promote_types(
        sys1.P.sys.A.dtype if sys1.P.sys.A.size > 0 else float,
        sys2.P.sys.A.dtype if sys2.P.sys.A.size > 0 else float
    )
    # Promote delay types
    S = np.promote_types(
        np.asarray(sys1.tau).dtype if len(sys1.tau) > 0 else float,
        np.asarray(sys2.tau).dtype if len(sys2.tau) > 0 else float
    )
    return T, S


def _convert_to_delay_lti(sys):
    """Convert a system to a DelayLTISystem if necessary."""
    if isinstance(sys, DelayLTISystem):
        return sys
    elif isinstance(sys, StateSpace):
        return DelayLTISystem.from_ss(sys)
    elif isinstance(sys, TransferFunction):
        return tf2dlti(sys)
    else:
        raise TypeError("Unsupported system type for DelayLTISystem conversion: {}".format(type(sys)))
    

def vcat(*systems: list[DelayLTISystem]) -> DelayLTISystem:
    from .partitionedssp import vcat_pss

    if not all(isinstance(sys, DelayLTISystem) for sys in systems):
        raise TypeError("All inputs must be DelayLTISystems")

    part_ssp = [sys.P for sys in systems]
    P = vcat_pss(*part_ssp)
    tau = np.concatenate([sys.tau for sys in systems])
    return DelayLTISystem(P, tau)
    

def hcat(*systems: list[DelayLTISystem]) -> DelayLTISystem:
    from .partitionedssp import hcat_pss

    if not(all(isinstance(sys, DelayLTISystem) for sys in systems)):
        raise TypeError("All inputs must be DelayLTISystems")
    
    part_ssp = [sys.P for sys in systems]
    P = hcat_pss(*part_ssp)
    tau = np.concatenate([sys.tau for sys in systems])
    return DelayLTISystem(P, tau)
    

def mimo_delay(array: np.ndarray[DelayLTISystem]):
    rows = [hcat(*row) for row in array]
    return vcat(*rows)