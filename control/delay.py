# delay.py - functions involving time delays
#
# Initial author: Sawyer Fuller
# Creation date: 26 Aug 2010

"""Functions to implement time delays (pade)."""

__all__ = ['pade']

def pade(T, n=1, numdeg=None):
    """Create a linear system that approximates a delay.

    Return the numerator and denominator coefficients of the Pade
    approximation of the given order.

    Parameters
    ----------
    T : number
        Time. delay
    n : positive integer
        Degree of denominator of approximation.
    numdeg : integer, or None (the default)
        If numdeg is None, numerator degree equals denominator degree.
        If numdeg >= 0, specifies degree of numerator.
        If numdeg < 0, numerator degree is n+numdeg.

    Returns
    -------
    num, den : ndarray
        Polynomial coefficients of the delay model, in descending powers of s.

    Notes
    -----
    Based on [1]_ and [2]_.

    References
    ----------
    .. [1] Algorithm 11.3.1 in Golub and van Loan, "Matrix Computation" 3rd.
         Ed. pp. 572-574.

    .. [2] M. Vajta, "Some remarks on Padé-approximations",
         3rd TEMPUS-INTCOM Symposium.

    Examples
    --------
    >>> delay = 1
    >>> num, den = ct.pade(delay, 3)
    >>> num, den
    ([-1.0, 12.0, -60.0, 120.0], [1.0, 12.0, 60.0, 120.0])

    >>> num, den = ct.pade(delay, 3, -2)
    >>> num, den
    ([-6.0, 24.0], [1.0, 6.0, 18.0, 24.0])

    """
    if numdeg is None:
        numdeg = n
    elif numdeg < 0:
        numdeg += n

    if not T >= 0:
        raise ValueError("require T >= 0")
    if not n >= 0:
        raise ValueError("require n >= 0")
    if not (0 <= numdeg <= n):
        raise ValueError("require 0 <= numdeg <= n")

    if T == 0:
        num = [1,]
        den = [1,]
    else:
        num = [0. for i in range(numdeg+1)]
        num[-1] = 1.
        cn = 1.
        for k in range(1, numdeg+1):
            # derived from Golub and van Loan eq. for Dpq(z) on p. 572
            # this accumulative style follows Alg 11.3.1
            cn *= -T * (numdeg - k + 1)/(numdeg + n - k + 1)/k
            num[numdeg-k] = cn

        den = [0. for i in range(n+1)]
        den[-1] = 1.
        cd = 1.
        for k in range(1, n+1):
            # see cn above
            cd *= T * (n - k + 1)/(numdeg + n - k + 1)/k
            den[n-k] = cd

        num = [coeff/den[0] for coeff in num]
        den = [coeff/den[0] for coeff in den]
    return num, den
