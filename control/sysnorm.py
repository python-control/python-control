# -*- coding: utf-8 -*-
"""sysnorm.py

Functions for computing system norms.

Routine in this module:

norm

Created on Thu Dec 21 08:06:12 2023
Author: Henrik Sandberg
"""

import numpy as np
import numpy.linalg as la

import control as ct

__all__ = ['norm']

#------------------------------------------------------------------------------

def norm(system, p=2, tol=1e-6, print_warning=True):
    """Computes norm of system.
    
    Parameters
    ----------
    system : LTI (:class:`StateSpace` or :class:`TransferFunction`)
        System in continuous or discrete time for which the norm should be computed.
    p : int or str
        Type of norm to be computed. p=2 gives the H_2 norm, and p='inf' gives the L_infinity norm.
    tol : float
        Relative tolerance for accuracy of L_infinity norm computation. Ignored
        unless p='inf'.
    print_warning : bool
        Print warning message in case norm value may be uncertain.
    
    Returns
    -------
    norm_value : float or NoneType
        Norm value of system (float) or None if computation could not be completed.
   
    Notes
    -----
    Does not yet compute the L_infinity norm for discrete time systems with pole(s) in z=0.
    
    Examples
    --------
    >>> Gc = ct.tf([1], [1, 2, 1])
    >>> ct.norm(Gc,2)
    0.5000000000000001
    >>> ct.norm(Gc,'inf',tol=1e-10)
    1.0000000000582077
    """
    G = ct.ss(system)
    A = G.A
    B = G.B
    C = G.C
    D = G.D
    
    #
    # H_2-norm computation
    #
    if p == 2:  
        # Continuous time case
        if G.isctime():
            poles_real_part = G.poles().real    
            if any(np.isclose(poles_real_part, 0.0)):
                if print_warning:
                    print("Warning: Poles close to, or on, the imaginary axis. Norm value may be uncertain.")
                return float('inf')
            elif (D != 0).any() or any(poles_real_part > 0.0):  # System unstable or has direct feedthrough?
                return float('inf')
            else:
                try:
                    P = ct.lyap(A, B@B.T)
                except Exception as e:
                    print(f"An error occurred solving the continuous time Lyapunov equation: {e}")
                    return None
                
                # System is stable to reach this point, and P should be positive semi-definite. 
                # Test next is a precaution in case the Lyapunov equation is ill conditioned.
                if any(la.eigvals(P) < 0.0):  
                    if print_warning:
                        print("Warning: There appears to be poles close to the imaginary axis. Norm value may be uncertain.")
                    return float('inf')
                else:
                    norm_value = np.sqrt(np.trace(C@P@C.T))  # Argument in sqrt should be non-negative
                    if np.isnan(norm_value):
                        print("Unknown error. Norm computation resulted in NaN.")           
                        return None
                    else:
                        return norm_value
        
        # Discrete time case
        elif G.isdtime():
            poles_abs = abs(G.poles())
            if any(np.isclose(poles_abs, 1.0)):
                if print_warning:
                    print("Warning: Poles close to, or on, the complex unit circle. Norm value may be uncertain.")
                return float('inf')
            elif any(poles_abs > 1.0):  # System unstable?
                return float('inf')
            else:
                try:
                    P = ct.dlyap(A, B@B.T)
                except Exception as e:
                    print(f"An error occurred solving the discrete time Lyapunov equation: {e}")
                    return None
                
                # System is stable to reach this point, and P should be positive semi-definite. 
                # Test next is a precaution in case the Lyapunov equation is ill conditioned.
                if any(la.eigvals(P) < 0.0):
                    if print_warning:
                        print("Warning: There appears to be poles close to the complex unit circle. Norm value may be uncertain.")
                    return float('inf')
                else:
                    norm_value = np.sqrt(np.trace(C@P@C.T + D@D.T))  # Argument in sqrt should be non-negative               
                    if np.isnan(norm_value):
                        print("Unknown error. Norm computation resulted in NaN.")           
                        return None
                    else:
                        return norm_value               
    #
    # L_infinity-norm computation
    #
    elif p == "inf":   
        def _Hamilton_matrix(gamma):
            """Constructs Hamiltonian matrix. For internal use."""
            R = Ip*gamma**2 - D.T@D
            invR = la.inv(R)
            return np.block([[A+B@invR@D.T@C, B@invR@B.T], [-C.T@(Ip+D@invR@D.T)@C, -(A+B@invR@D.T@C).T]])    
    
        # Discrete time case 
        # Use inverse bilinear transformation of discrete time system to s-plane if no poles on |z|=1 or z=0.
        # Allows us to use test for continuous time systems next.
        if G.isdtime():
            Ad = A
            Bd = B
            Cd = C
            Dd = D
            if any(np.isclose(abs(la.eigvals(Ad)), 1.0)):
                if print_warning:
                    print("Warning: Poles close to, or on, the complex unit circle. Norm value may be uncertain.")
                return float('inf')
            elif any(np.isclose(la.eigvals(Ad), 0.0)):
                print("L_infinity norm computation for discrete time system with pole(s) at z=0 currently not supported.")            
                return None
            
            # Inverse bilinear transformation
            In = np.eye(len(Ad))
            Adinv = la.inv(Ad+In)
            A = 2*(Ad-In)@Adinv
            B = 2*Adinv@Bd
            C = 2*Cd@Adinv
            D = Dd - Cd@Adinv@Bd
        
        # Continus time case 
        if any(np.isclose(la.eigvals(A).real, 0.0)):
            if print_warning:
                print("Warning: Poles close to, or on, imaginary axis. Norm value may be uncertain.")
            return float('inf')
    
        gaml = la.norm(D,ord=2)    # Lower bound
        gamu = max(1.0, 2.0*gaml)  # Candidate upper bound
        Ip = np.eye(len(D))    
         
        while any(np.isclose(la.eigvals(_Hamilton_matrix(gamu)).real, 0.0)):  # Find actual upper bound
            gamu *= 2.0
        
        while (gamu-gaml)/gamu > tol:
            gam = (gamu+gaml)/2.0
            if any(np.isclose(la.eigvals(_Hamilton_matrix(gam)).real, 0.0)):
                gaml = gam
            else:
                gamu = gam
        return gam
    else:
        print(f"Norm computation for p={p} currently not supported.")           
        return None
