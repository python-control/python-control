# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 08:06:12 2023

@author: hsan
"""

import numpy as np
import numpy.linalg as la

import control as ct

#------------------------------------------------------------------------------

def norm(system, p=2, tol=1e-6):
    """Computes H_2 (p=2) or L_infinity (p="inf", tolerance tol) norm of system."""
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
        def Hamilton_matrix(gamma):
            """Constructs Hamiltonian matrix."""
            R = Ip*gamma**2 - D.T@D
            invR = la.inv(R)
            return np.block([[A+B@invR@D.T@C, B@invR@B.T], [-C.T@(Ip+D@invR@D.T)@C, -(A+B@invR@D.T@C).T]])    
    
        if G.isdtime(): # Bilinear transformation to s-plane
            Ad = A
            Bd = B
            Cd = C
            Dd = D
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
         
        while any(np.isclose(la.eigvals(Hamilton_matrix(gamu)).real, 0.0)): # Find an upper bound
            gamu *= 2.0
        
        while (gamu-gaml)/gamu > tol:
            gam = (gamu+gaml)/2.0
            if any(np.isclose(la.eigvals(Hamilton_matrix(gam)).real, 0.0)):
                gaml = gam
            else:
                gamu = gam
        return gam
    else:
        # Norm computation only supported for p=2 and p='inf'
        return None
