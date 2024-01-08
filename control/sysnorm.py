# -*- coding: utf-8 -*-
"""sysnorm.py

Functions for computing system norms.

Routines in this module:

norm()

Created on Thu Dec 21 08:06:12 2023
Author: Henrik Sandberg
"""

import numpy as np
import numpy.linalg as la

import control as ct

#------------------------------------------------------------------------------

def norm(system, p=2, tol=1e-6):
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
    
    Returns
    -------
    norm : float
        Norm of system
   
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
    
    if p == 2:  # H_2-norm
        if G.isctime():
            if (D != 0).any() or any(G.poles().real >= 0):
                return float('inf')
            else:
                P = ct.lyap(A, B@B.T)
                return np.sqrt(np.trace(C@P@C.T))
        elif G.isdtime():
            if any(abs(G.poles()) >= 1):
                return float('inf')
            else:
                P = ct.dlyap(A, B@B.T)
                return np.sqrt(np.trace(C@P@C.T + D@D.T))
   
    elif p == "inf":    # L_infinity-norm
        def _Hamilton_matrix(gamma):
            """Constructs Hamiltonian matrix."""
            R = Ip*gamma**2 - D.T@D
            invR = la.inv(R)
            return np.block([[A+B@invR@D.T@C, B@invR@B.T], [-C.T@(Ip+D@invR@D.T)@C, -(A+B@invR@D.T@C).T]])    
    
        if G.isdtime(): # Bilinear transformation of discrete time system to s-plane if no poles at |z|=1 or z=0
            Ad = A
            Bd = B
            Cd = C
            Dd = D
            if any(np.isclose(abs(la.eigvals(Ad)), 1.0)):
                return float('inf')
            elif any(np.isclose(la.eigvals(Ad), 0.0)):
                print("L_infinity norm computation for discrete time system with pole(s) at z = 0 currently not supported.")            
                return None
            In = np.eye(len(Ad))
            Adinv = la.inv(Ad+In)
            A = 2*(Ad-In)@Adinv
            B = 2*Adinv@Bd
            C = 2*Cd@Adinv
            D = Dd - Cd@Adinv@Bd
               
        if any(np.isclose(la.eigvals(A).real, 0.0)):
            return float('inf')
    
        gaml = la.norm(D,ord=2) # Lower bound
        gamu = max(1.0, 2.0*gaml) # Candidate upper bound
        Ip = np.eye(len(D))    
         
        while any(np.isclose(la.eigvals(_Hamilton_matrix(gamu)).real, 0.0)): # Find an upper bound
            gamu *= 2.0
        
        while (gamu-gaml)/gamu > tol:
            gam = (gamu+gaml)/2.0
            if any(np.isclose(la.eigvals(_Hamilton_matrix(gam)).real, 0.0)):
                gaml = gam
            else:
                gamu = gam
        return gam
    else:
        print("Norm computation for p =", p, "currently not supported.")           
        return None
