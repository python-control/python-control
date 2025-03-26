import numpy as np
from .lti import LTI
from .partitionedssp import PartitionedStateSpace
from .statesp import ss, StateSpace, tf2ss
from .xferfcn import TransferFunction
from .iosys import _process_iosys_keywords



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
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))
        
    def __rmul__(self, other):  # Handle  System * Number
        if isinstance(other, (int, float, complex)):
            new_B = np.block([self.P.B1 * other, self.P.B2])
            new_D = np.block([[self.P.D11 * other, self.P.D12], [self.P.D21 * other, self.P.D22]])
            new_P = PartitionedStateSpace(ss(self.P.A, new_B, self.P.C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTISystem(new_P, self.tau)
        
        elif isinstance(other, TransferFunction):
            dlti = tf2dlti(other)
            return dlti * self
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(other), type(self)))

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            # Add to direct term from input to output
            new_D = self.P.sys.D.copy()
            new_D[:self.noutputs(), :self.ninputs()] += other
            pnew = PartitionedStateSpace(ss(self.P.sys.A, self.P.sys.B, self.P.sys.C, new_D), self.P.nu1, self.P.ny1)
            return DelayLTISystem(pnew, self.tau)
        
        elif isinstance(other, DelayLTISystem):
            psys_new = self.P + other.P
            tau_new = np.concatenate([self.tau, other.tau])
            return DelayLTISystem(psys_new.P, tau_new)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        return -self + other

    def __eq__(self, other):
        if not isinstance(other, DelayLTISystem):
            return False
        return (np.array_equal(self.P.A, other.P.A) and
                np.array_equal(self.P.B, other.P.B) and
                np.array_equal(self.P.C, other.P.C) and
                np.array_equal(self.P.D, other.P.D) and
                np.array_equal(self.tau, other.tau))
        
    def feedback(self, other):
            psys_new = self.P.feedback(other.P)
            tau_new  = np.concatenate([self.tau, other.tau])
            return DelayLTISystem(psys_new.P, tau_new)
    
    def __str__(self):
        s = "DelayLTISystem\n"
        s += "P:\n" + str(self.P) + "\n"
        s += "A =\n" + str(self.P.A) + "\n"
        s += "B =\n" + str(self.P.B) + "\n"
        s += "C =\n" + str(self.P.C) + "\n"
        s += "D =\n" + str(self.P.D) + "\n" + "\n"
        s += "delays: " + str(self.tau)
        return s


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

    if hasattr(tf, 'delays'):
        tau = tf.delays
        delay_tau= delay(tau)
        return DelayLTISystem.from_ss(ss_tf) * delay_tau
    else:
        return DelayLTISystem.from_ss(ss_tf)