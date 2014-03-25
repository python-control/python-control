""" mateqn.py 

Matrix equation solvers (Lyapunov, Riccati)

Implementation of the functions lyap, dlyap, care and dare
for solution of Lyapunov and Riccati equations. """

# Python 3 compatability (needs to go here)
from __future__ import print_function

"""Copyright (c) 2011, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the project author nor the names of its 
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Bjorn Olofsson
"""

from numpy.linalg import inv
from scipy import shape, size, asarray, copy, zeros, eye, dot
from control.exception import ControlSlycot, ControlArgument

#### Lyapunov equation solvers lyap and dlyap

def lyap(A,Q,C=None,E=None):
    """ X = lyap(A,Q) solves the continuous-time Lyapunov equation
    
        A X + X A^T + Q = 0

    where A and Q are square matrices of the same dimension. 
    Further, Q must be symmetric.

    X = lyap(A,Q,C) solves the Sylvester equation

        A X + X Q + C = 0

    where A and Q are square matrices.

    X = lyap(A,Q,None,E) solves the generalized continuous-time
    Lyapunov equation

        A X E^T + E X A^T + Q = 0

    where Q is a symmetric matrix and A, Q and E are square matrices
    of the same dimension. """

    # Make sure we have access to the right slycot routines
    try:
        from slycot import sb03md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb03md'")

    try:
        from slycot import sb04md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb04md'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1,A.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1,Q.size)

    if C != None and len(shape(C)) == 1:
        C = C.reshape(1,C.size)

    if E != None and len(shape(E)) == 1:
        E = E.reshape(1,E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A,0)

    if size(Q) == 1:
        m = 1
    else:
        m = size(Q,0)

    # Solve standard Lyapunov equation
    if C==None and E==None:
        # Check input data for consistency
        if shape(A) != shape(Q):
            raise ControlArgument("A and Q must be matrices of identical \
                                sizes.")

        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if size(Q) > 1 and shape(Q)[0] != shape(Q)[1]:
            raise ControlArgument("Q must be a quadratic matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        # Solve the Lyapunov equation by calling Slycot function sb03md
        try:
            X,scale,sep,ferr,w = sb03md(n,-Q,A,eye(n,n),'C',trana='T')
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == n+1:
                e = ValueError("The matrix A and -A have common or very \
                    close eigenvalues.")
                e.info = ve.info
            else:
                e = ValueError("The QR algorithm failed to compute all \
                    the eigenvalues (see LAPACK Library routine DGEES).")
                e.info = ve.info
            raise e

    # Solve the Sylvester equation
    elif C != None and E==None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if size(Q) > 1 and shape(Q)[0] != shape(Q)[1]:
            raise ControlArgument("Q must be a quadratic matrix.")

        if (size(C) > 1 and shape(C)[0] != n) or \
            (size(C) > 1 and shape(C)[1] != m) or \
            (size(C) == 1 and size(A) != 1) or (size(C) == 1 and size(Q) != 1):
            raise ControlArgument("C matrix has incompatible dimensions.")

        # Solve the Sylvester equation by calling the Slycot function sb04md
        try:
            X = sb04md(n,m,A,Q,-C)
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info > m:
                e = ValueError("A singular matrix was encountered whilst \
                    solving for the %i-th column of matrix X." % ve.info-m)
                e.info = ve.info
            else:
                e = ValueError("The QR algorithm failed to compute all the \
                    eigenvalues (see LAPACK Library routine DGEES).")
                e.info = ve.info
            raise e

    # Solve the generalized Lyapunov equation
    elif C == None and E != None:
        # Check input data for consistency
        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            (size(Q) == 1 and n > 1):
            raise ControlArgument("Q must be a square matrix with the same \
                dimension as A.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
            (size(E) > 1 and shape(E)[0] != n) or \
            (size(E) == 1 and n > 1):
            raise ControlArgument("E must be a square matrix with the same \
                dimension as A.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        # Make sure we have access to the write slicot routine
        try:
            from slycot import sg03ad
        except ImportError:
            raise ControlSlycot("can't find slycot module 'sg03ad'")

        # Solve the generalized Lyapunov equation by calling Slycot 
        # function sg03ad
        try:
            A,E,Q,Z,X,scale,sep,ferr,alphar,alphai,beta = \
                sg03ad('C','B','N','T','L',n,A,E,eye(n,n),eye(n,n),-Q)
        except ValueError(ve):
            if ve.info < 0 or ve.info > 4:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix contained in the upper \
                                Hessenberg part of the array A is not in \
                                upper quasitriangular form")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The pencil A - lambda * E cannot be \
                                reduced to generalized Schur form: LAPACK \
                                routine DGEGS has failed to converge")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("The pencil A - lambda * E has a \
                                degenerate pair of eigenvalues. That is, \
                                lambda_i = lambda_j for some i and j, where \
                                lambda_i and lambda_j are eigenvalues of \
                                A - lambda * E. Hence, the equation is \
                                singular;  perturbed values were \
                                used to solve the equation (but the matrices \
                                A and E are unchanged)")
                e.info = ve.info
            raise e    
    # Invalid set of input parameters    
    else:
        raise ControlArgument("Invalid set of input parameters")
            
    return X


def dlyap(A,Q,C=None,E=None):
    """ dlyap(A,Q) solves the discrete-time Lyapunov equation

        A X A^T - X + Q = 0

    where A and Q are square matrices of the same dimension. Further
    Q must be symmetric.

    dlyap(A,Q,C) solves the Sylvester equation

        A X Q^T - X + C = 0

    where A and Q are square matrices.

    dlyap(A,Q,None,E) solves the generalized discrete-time Lyapunov
    equation

        A X A^T - E X E^T + Q = 0

    where Q is a symmetric matrix and A, Q and E are square matrices
    of the same dimension. """

    # Make sure we have access to the right slycot routines
    try:
        from slycot import sb03md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb03md'")

    try:
        from slycot import sb04qd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb04qd'")

    try:
        from slycot import sg03ad
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sg03ad'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1,A.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1,Q.size)

    if C != None and len(shape(C)) == 1:
        C = C.reshape(1,C.size)

    if E != None and len(shape(E)) == 1:
        E = E.reshape(1,E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A,0)

    if size(Q) == 1:
        m = 1
    else:
        m = size(Q,0)

    # Solve standard Lyapunov equation
    if C==None and E==None:
        # Check input data for consistency
        if shape(A) != shape(Q):
            raise ControlArgument("A and Q must be matrices of identical \
                                 sizes.")

        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if size(Q) > 1 and shape(Q)[0] != shape(Q)[1]:
            raise ControlArgument("Q must be a quadratic matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        # Solve the Lyapunov equation by calling the Slycot function sb03md
        try:
            X,scale,sep,ferr,w = sb03md(n,-Q,A,eye(n,n),'D',trana='T')
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            else:
                e = ValueError("The QR algorithm failed to compute all the \
                    eigenvalues (see LAPACK Library routine DGEES).")
                e.info = ve.info
            raise e

    # Solve the Sylvester equation
    elif C != None and E==None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix")

        if size(Q) > 1 and shape(Q)[0] != shape(Q)[1]:
            raise ControlArgument("Q must be a quadratic matrix")

        if (size(C) > 1 and shape(C)[0] != n) or \
            (size(C) > 1 and shape(C)[1] != m) or \
            (size(C) == 1 and size(A) != 1) or (size(C) == 1 and size(Q) != 1):
            raise ControlArgument("C matrix has incompatible dimensions")

        # Solve the Sylvester equation by calling Slycot function sb04qd
        try:
            X = sb04qd(n,m,-A,asarray(Q).T,C)
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info > m:
                e = ValueError("A singular matrix was encountered whilst \
                    solving for the %i-th column of matrix X." % ve.info-m)
                e.info = ve.info
            else:
                e = ValueError("The QR algorithm failed to compute all the \
                    eigenvalues (see LAPACK Library routine DGEES)")
                e.info = ve.info
            raise e

    # Solve the generalized Lyapunov equation
    elif C == None and E != None:
        # Check input data for consistency
        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            (size(Q) == 1 and n > 1):
            raise ControlArgument("Q must be a square matrix with the same \
                dimension as A.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
            (size(E) > 1 and shape(E)[0] != n) or \
            (size(E) == 1 and n > 1):
            raise ControlArgument("E must be a square matrix with the same \
                dimension as A.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        # Solve the generalized Lyapunov equation by calling Slycot 
        # function sg03ad
        try:
            A,E,Q,Z,X,scale,sep,ferr,alphar,alphai,beta = \
                sg03ad('D','B','N','T','L',n,A,E,eye(n,n),eye(n,n),-Q)
        except ValueError(ve):
            if ve.info < 0 or ve.info > 4:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix contained in the upper \
                                Hessenberg part of the array A is not in \
                                upper quasitriangular form")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The pencil A - lambda * E cannot be \
                                reduced to generalized Schur form: LAPACK \
                                routine DGEGS has failed to converge")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("The pencil A - lambda * E has a \
                                pair of reciprocal eigenvalues. That is, \
                                lambda_i = 1/lambda_j for some i and j, \
                                where  lambda_i and lambda_j are eigenvalues \
                                of A - lambda * E. Hence, the equation is \
                                singular;  perturbed values were \
                                used to solve the equation (but the \
                                matrices A and E are unchanged)")
                e.info = ve.info
            raise e
    # Invalid set of input parameters    
    else:
        raise ControlArgument("Invalid set of input parameters")

    return X



#### Riccati equation solvers care and dare

def care(A,B,Q,R=None,S=None,E=None):
    """ (X,L,G) = care(A,B,Q) solves the continuous-time algebraic Riccati
    equation

        A^T X + X A - X B B^T X + Q = 0

    where A and Q are square matrices of the same dimension. Further, Q 
    is a symmetric matrix. The function returns the solution X, the gain
    matrix G = B^T X and the closed loop eigenvalues L, i.e., the eigenvalues
    of A - B G.

    (X,L,G) = care(A,B,Q,R,S,E) solves the generalized continuous-time
    algebraic Riccati equation

        A^T X E + E^T X A - (E^T X B + S) R^-1 (B^T X E + S^T) + Q = 0

    where A, Q and E are square matrices of the same dimension. Further, Q and 
    R are symmetric matrices. The function returns the solution X, the gain
    matrix G = R^-1 (B^T X E + S^T) and the closed loop eigenvalues L, i.e., 
    the eigenvalues of A - B G , E. """

    # Make sure we can import required slycot routine
    try:
        from slycot import sb02md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md'")

    try:
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02mt'")

    # Make sure we can find the required slycot routine
    try:
        from slycot import sg02ad
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sg02ad'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1,A.size)

    if len(shape(B)) == 1:
        B = B.reshape(1,B.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1,Q.size)

    if R != None and len(shape(R)) == 1:
        R = R.reshape(1,R.size)

    if S != None and len(shape(S)) == 1:
        S = S.reshape(1,S.size)

    if E != None and len(shape(E)) == 1:
        E = E.reshape(1,E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A,0)

    if size(B) == 1:
        m = 1
    else:
        m = size(B,1)
    if R==None:
        R = eye(m,m)    

    # Solve the standard algebraic Riccati equation
    if S==None and E==None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
            size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        R_ba = copy(R)
        B_ba = copy(B)

        # Solve the standard algebraic Riccati equation by calling Slycot 
        # functions sb02mt and sb02md
        try:
            A_b,B_b,Q_b,R_b,L_b,ipiv,oufact,G = sb02mt(n,m,B,R)
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == m+1:
                e = ValueError("The matrix R is numerically singular.")
                e.info = ve.info
            else:
                e = ValueError("The %i-th element of d in the UdU (LdL) \
                    factorization is zero." % ve.info)
                e.info = ve.info
            raise e

        try:
            X,rcond,w,S_o,U,A_inv = sb02md(n,A,G,Q,'C')
        except ValueError(ve):
            if ve.info < 0 or ve.info > 5:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix A is (numerically) singular in \
                    discrete-time case.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The Hamiltonian or symplectic matrix H cannot \
                    be reduced to real Schur form.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("The real Schur form of the Hamiltonian or \
                    symplectic matrix H cannot be appropriately ordered.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("The Hamiltonian or symplectic matrix H has \
                    less than n stable eigenvalues.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The N-th order system of linear algebraic \
                         equations is singular to working precision.")
                e.info = ve.info
            raise e

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(dot(1/(R_ba),asarray(B_ba).T) , X)
        else:
            G = dot(dot(inv(R_ba),asarray(B_ba).T) , X)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X , w[:n] , G )

    # Solve the generalized algebraic Riccati equation
    elif S != None and E != None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
            size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
            (size(E) > 1 and shape(E)[0] != n) or \
            size(E) == 1 and n > 1:
            raise ControlArgument("E must be a quadratic matrix of the same \
                dimension as A.")

        if (size(R) > 1 and shape(R)[0] != shape(R)[1]) or \
            (size(R) > 1 and shape(R)[0] != m) or \
            size(R) == 1 and m > 1:
            raise ControlArgument("R must be a quadratic matrix of the same \
                dimension as the number of columns in the B matrix.")

        if (size(S) > 1 and shape(S)[0] != n) or \
            (size(S) > 1 and shape(S)[1] != m) or \
            size(S) == 1 and n > 1 or \
            size(S) == 1 and m > 1:
            raise ControlArgument("Incompatible dimensions of S matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        R_b = copy(R)
        B_b = copy(B)
        E_b = copy(E)
        S_b = copy(S)

        # Solve the generalized algebraic Riccati equation by calling the 
        # Slycot function sg02ad
        try:
            rcondu,X,alfar,alfai,beta,S_o,T,U,iwarn = \
                    sg02ad('C','B','N','U','N','N','S','R',n,m,0,A,E,B,Q,R,S)
        except ValueError(ve):
            if ve.info < 0 or ve.info > 7:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The computed extended matrix pencil is \
                            singular, possibly due to rounding errors.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The QZ algorithm failed.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("Reordering of the generalized eigenvalues \
                    failed.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("After reordering, roundoff changed values of \
                            some complex eigenvalues so that leading \
                            eigenvalues in the generalized Schur form no \
                            longer satisfy the stability condition; this \
                            could also be caused due to scaling.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The computed dimension of the solution does \
                            not equal N.")
                e.info = ve.info
            elif ve.info == 6:
                e = ValueError("The spectrum is too close to the boundary of \
                            the stability domain.")
                e.info = ve.info
            elif ve.info == 7:
                e = ValueError("A singular matrix was encountered during the \
                            computation of the solution matrix X.")
                e.info = ve.info
            raise e

        # Calculate the closed-loop eigenvalues L
        L = zeros((n,1))
        L.dtype = 'complex64'
        for i in range(n):
            L[i] = (alfar[i] + alfai[i]*1j)/beta[i]

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot(1/(R_b),dot(asarray(B_b).T,dot(X,E_b))+asarray(S_b).T)
        else:
            G = dot(inv(R_b),dot(asarray(B_b).T,dot(X,E_b))+asarray(S_b).T)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X , L , G)
    
    # Invalid set of input parameters
    else:
        raise ControlArgument("Invalid set of input parameters.")


def dare(A,B,Q,R,S=None,E=None):
    """ (X,L,G) = dare(A,B,Q,R) solves the discrete-time algebraic Riccati  
    equation

        A^T X A - X - A^T X B (B^T X B + R)^-1 B^T X A + Q = 0

    where A and Q are square matrices of the same dimension. Further, Q 
    is a symmetric matrix. The function returns the solution X, the gain
    matrix G = (B^T X B + R)^-1 B^T X A and the closed loop eigenvalues L,
    i.e., the eigenvalues of A - B G.

    (X,L,G) = dare(A,B,Q,R,S,E) solves the generalized discrete-time algebraic
    Riccati equation

        A^T X A - E^T X E - (A^T X B + S) (B^T X B + R)^-1 (B^T X A + S^T) + 
            + Q = 0

    where A, Q and E are square matrices of the same dimension. Further, Q and 
    R are symmetric matrices. The function returns the solution X, the gain
    matrix G = (B^T X B + R)^-1 (B^T X A + S^T) and the closed loop
    eigenvalues L, i.e., the eigenvalues of A - B G , E. """

    # Make sure we can import required slycot routine
    try:
        from slycot import sb02md
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md'")

    try:
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02mt'")

    # Make sure we can find the required slycot routine
    try:
        from slycot import sg02ad
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sg02ad'")

    # Reshape 1-d arrays
    if len(shape(A)) == 1:
        A = A.reshape(1,A.size)

    if len(shape(B)) == 1:
        B = B.reshape(1,B.size)

    if len(shape(Q)) == 1:
        Q = Q.reshape(1,Q.size)

    if R != None and len(shape(R)) == 1:
        R = R.reshape(1,R.size)

    if S != None and len(shape(S)) == 1:
        S = S.reshape(1,S.size)

    if E != None and len(shape(E)) == 1:
        E = E.reshape(1,E.size)

    # Determine main dimensions
    if size(A) == 1:
        n = 1
    else:
        n = size(A,0)

    if size(B) == 1:
        m = 1
    else:
        m = size(B,1)

    # Solve the standard algebraic Riccati equation
    if S==None and E==None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
            size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        A_ba = copy(A)
        R_ba = copy(R)
        B_ba = copy(B)

        # Solve the standard algebraic Riccati equation by calling Slycot 
        # functions sb02mt and sb02md
        try:
            A_b,B_b,Q_b,R_b,L_b,ipiv,oufact,G = sb02mt(n,m,B,R)    
        except ValueError(ve):
            if ve.info < 0:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == m+1:
                e = ValueError("The matrix R is numerically singular.")
                e.info = ve.info
            else:
                e = ValueError("The %i-th element of d in the UdU (LdL) \
                     factorization is zero." % ve.info)
                e.info = ve.info
            raise e

        try:
            X,rcond,w,S,U,A_inv = sb02md(n,A,G,Q,'D')
        except ValueError(ve):
            if ve.info < 0 or ve.info > 5:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The matrix A is (numerically) singular in \
                    discrete-time case.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The Hamiltonian or symplectic matrix H cannot \
                    be reduced to real Schur form.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("The real Schur form of the Hamiltonian or \
                     symplectic matrix H cannot be appropriately ordered.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("The Hamiltonian or symplectic matrix H has \
                     less than n stable eigenvalues.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The N-th order system of linear algebraic \
                     equations is singular to working precision.")
                e.info = ve.info
            raise e

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot( 1/(dot(asarray(B_ba).T,dot(X,B_ba))+R_ba) , \
                dot(asarray(B_ba).T,dot(X,A_ba)) )
        else:
            G = dot( inv(dot(asarray(B_ba).T,dot(X,B_ba))+R_ba) , \
                dot(asarray(B_ba).T,dot(X,A_ba)) )

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X , w[:n] , G)

    # Solve the generalized algebraic Riccati equation
    elif S != None and E != None:
        # Check input data for consistency
        if size(A) > 1 and shape(A)[0] != shape(A)[1]:
            raise ControlArgument("A must be a quadratic matrix.")

        if (size(Q) > 1 and shape(Q)[0] != shape(Q)[1]) or \
            (size(Q) > 1 and shape(Q)[0] != n) or \
            size(Q) == 1 and n > 1:
            raise ControlArgument("Q must be a quadratic matrix of the same \
                dimension as A.")

        if (size(B) > 1 and shape(B)[0] != n) or \
            size(B) == 1 and n > 1:
            raise ControlArgument("Incompatible dimensions of B matrix.")

        if (size(E) > 1 and shape(E)[0] != shape(E)[1]) or \
            (size(E) > 1 and shape(E)[0] != n) or \
            size(E) == 1 and n > 1:
            raise ControlArgument("E must be a quadratic matrix of the same \
                dimension as A.")

        if (size(R) > 1 and shape(R)[0] != shape(R)[1]) or \
            (size(R) > 1 and shape(R)[0] != m) or \
            size(R) == 1 and m > 1:
            raise ControlArgument("R must be a quadratic matrix of the same \
                dimension as the number of columns in the B matrix.")

        if (size(S) > 1 and shape(S)[0] != n) or \
            (size(S) > 1 and shape(S)[1] != m) or \
            size(S) == 1 and n > 1 or \
            size(S) == 1 and m > 1:
            raise ControlArgument("Incompatible dimensions of S matrix.")

        if not (asarray(Q) == asarray(Q).T).all():
            raise ControlArgument("Q must be a symmetric matrix.")

        if not (asarray(R) == asarray(R).T).all():
            raise ControlArgument("R must be a symmetric matrix.")

        # Create back-up of arrays needed for later computations
        A_b = copy(A)
        R_b = copy(R)
        B_b = copy(B)
        E_b = copy(E)
        S_b = copy(S)

        # Solve the generalized algebraic Riccati equation by calling the 
        # Slycot function sg02ad
        try:
            rcondu,X,alfar,alfai,beta,S_o,T,U,iwarn = \
                    sg02ad('D','B','N','U','N','N','S','R',n,m,0,A,E,B,Q,R,S)
        except ValueError(ve):
            if ve.info < 0 or ve.info > 7:
                e = ValueError(ve.message)
                e.info = ve.info
            elif ve.info == 1:
                e = ValueError("The computed extended matrix pencil is \
                            singular, possibly due to rounding errors.")
                e.info = ve.info
            elif ve.info == 2:
                e = ValueError("The QZ algorithm failed.")
                e.info = ve.info
            elif ve.info == 3:
                e = ValueError("Reordering of the generalized eigenvalues \
                     failed.")
                e.info = ve.info
            elif ve.info == 4:
                e = ValueError("After reordering, roundoff changed values of \
                            some complex eigenvalues so that leading \
                            eigenvalues in the generalized Schur form no \
                            longer satisfy the stability condition; this \
                            could also be caused due to scaling.")
                e.info = ve.info
            elif ve.info == 5:
                e = ValueError("The computed dimension of the solution does \
                            not equal N.")
                e.info = ve.info
            elif ve.info == 6:
                e = ValueError("The spectrum is too close to the boundary of \
                            the stability domain.")
                e.info = ve.info
            elif ve.info == 7:
                e = ValueError("A singular matrix was encountered during the \
                            computation of the solution matrix X.")
                e.info = ve.info
            raise e

        L = zeros((n,1))
        L.dtype = 'complex64'
        for i in range(n):
            L[i] = (alfar[i] + alfai[i]*1j)/beta[i]

        # Calculate the gain matrix G
        if size(R_b) == 1:
            G = dot( 1/(dot(asarray(B_b).T,dot(X,B_b))+R_b) , \
                dot(asarray(B_b).T,dot(X,A_b)) + asarray(S_b).T)
        else:
            G = dot( inv(dot(asarray(B_b).T,dot(X,B_b))+R_b) , \
                dot(asarray(B_b).T,dot(X,A_b)) + asarray(S_b).T)

        # Return the solution X, the closed-loop eigenvalues L and
        # the gain matrix G
        return (X , L , G)

    # Invalid set of input parameters
    else:
        raise ControlArgument("Invalid set of input parameters.")
