# controls.py - Ryan Krauss's control module
# $Id$

"""This module is for analyzing linear, time-invariant dynamic systems
and feedback control systems using the Laplace transform.  The heart
of the module is the TransferFunction class, which represents a
transfer function as a ratio of numerator and denominator polynomials
in s.  TransferFunction is derived from scipy.signal.lti."""

import glob, pdb
from math import atan2, log10

from scipy import *
from scipy.linalg import inv as inverse
from scipy.optimize import newton, fmin, fminbound
from scipy.io import read_array, save, loadmat, write_array
from scipy import signal

from  IPython.Debugger import Pdb

import sys, os, copy, time

from matplotlib.ticker import LogFormatterMathtext

version = '1.1.0'

class MyFormatter(LogFormatterMathtext):
   def __call__(self, x, pos=None):
       if pos==0: return ''  # pos=0 is the first tick
       else: return LogFormatterMathtext.__call__(self, x, pos)


def shift(vectin, new):
    N = len(vectin)-1
    for n in range(N,0,-1):
        vectin[n]=vectin[n-1]
    vectin[0]=new
    return vectin

def myeq(p1,p2):
    """Test the equality of the of two polynomials based on
    coeffiecents."""
    if hasattr(p1, 'coeffs') and hasattr(p2, 'coeffs'):
       c1=p1.coeffs
       c2=p2.coeffs
    else:
       return False
    if len(c1)!=len(c2):
        return False
    else:
        testvect=c1==c2
        if hasattr(testvect,'all'):
            return testvect.all()
        else:
            return testvect

def build_fit_matrix(output_vect, input_vect, numorder, denorder):
    """Build the [A] matrix used in least squares curve fitting
    according to

    output_vect = [A]c

    as described in fit_discrete_response."""
    A = zeros((len(output_vect),numorder+denorder+1))#the +1 accounts
         #for the fact that both the numerator and the denominator
         #have zero-order terms (which would give +2), but the
         #zero order denominator term is actually not used in the fit
         #(that is the output vector)
    curin = input_vect
    A[:,0] = curin
    for n in range(1, numorder+1):
        curin = r_[[0.0], curin[0:-1]]#prepend a 0 to curin and drop its
                                    #last element
        A[:,n] = curin
    curout = -output_vect#this is the first output column, but it not
                        #actually used
    firstden = numorder+1
    for n in range(0, denorder):
        curout = r_[[0.0], curout[0:-1]]
        A[:,firstden+n] = curout
    return A
    

def fit_discrete_response(output_vect, input_vect, numorder, denorder):
    """Find the coefficients of a digital transfer function that give
    the best fit to output_vect in a least squares sense.  output_vect
    is the output of the system and input_vect is the input.  The
    input and output vectors are shifted backward in time a maximum of
    numorder and denorder steps respectively.  Each shifted vector
    becomes a column in the matrix for the least squares curve fit of
    the form

    output_vect = [A]c

    where [A] is the matrix whose columns are shifted versions of
    input_vect and output_vect and c is composed of the numerator and
    denominator coefficients of the transfer function. numorder and
    denorder are the highest power of z in the numerator or
    denominator respectively.

    In essence, the approach is to find the coefficients that best fit
    related the input_vect and output_vect according to the difference
    equation

    y(k) = b_0 x(k) + b_1 x(k-1) + b_2 x(k-2) + ... + b_m x(k-m)
           - a_1 y(k-1) - a_2 y(k-2) - ... - a_n y(k-n)

    where x = input_vect, y = output_vect, m = numorder, and
    n = denorder.  The unknown coefficient vector is then

    c = [b_0, b_1, b_2, ... , b_m, a_1, a_2, ..., a_n]

    Note that a_0 is forced to be 1.

    The matrix [A] is then composed of [A] = [X(k) X(k-1) X(k-2)
    ... Y(k-1) Y(k-2) ...]  where X(k-2) represents the input_vect
    shifted 2 elements and Y(k-2) represents the output_vect shifted
    two elements."""
    A = build_fit_matrix(output_vect, input_vect, numorder, denorder)
    fitres = linalg.lstsq(A, output_vect)
    x = fitres[0]
    numz = x[0:numorder+1]
    denz = x[numorder+1:]
    denz = r_[[1.0],denz]
    return numz, denz

def prependzeros(num, den):
    nd = len(den)
    nn = len(num)
    if nn < nd:
        zvect = zeros(nd-nn)
        numout = r_[zvect, num]
    else:
        numout = num
    return numout, den

def in_with_tol(elem, searchlist, rtol=1e-5, atol=1e-10):
    """Determine whether or not elem+/-tol matches an element of
    searchlist."""
    for n, item in enumerate(searchlist):
       if allclose(item, elem, rtol=rtol, atol=atol):
            return n
    return -1



def PolyToLatex(polyin, var='s', fmt='%0.5g', eps=1e-12):
    N = polyin.order
    clist = polyin.coeffs
    outstr = ''
    for i, c in enumerate(clist):
        curexp = N-i
        curcoeff = fmt%c
        if curexp > 0:
            if curexp == 1:
                curs = var
            else:
                curs = var+'^%i'%curexp
            #Handle coeffs of +/- 1 in a special way:
            if 1-eps < c < 1+eps:
                curcoeff = ''
            elif -1-eps < c < -1+eps:
                curcoeff = '-'
        else:
            curs=''
        curstr = curcoeff+curs
        if c > 0 and outstr:
            curcoeff = '+'+curcoeff
        if abs(c) > eps:
            outstr+=curcoeff+curs
    return outstr

    
def polyfactor(num, den, prepend=True, rtol=1e-5, atol=1e-10):
    """Factor out any common roots from the polynomials represented by
    the vectors num and den and return new coefficient vectors with
    any common roots cancelled.

    Because poly1d does not think in terms of z^-1, z^-2, etc. it may
    be necessary to add zeros to the beginning of the numpoly coeffs
    to represent multiplying through be z^-n where n is the order of
    the denominator.  If prependzeros is Trus, the numerator and
    denominator coefficient vectors will have the same length."""
    numpoly = poly1d(num)
    denpoly = poly1d(den)
    nroots = roots(numpoly).tolist()
    droots = roots(denpoly).tolist()
    n = 0
    while n < len(nroots):
        curn = nroots[n]
        ind = in_with_tol(curn, droots, rtol=rtol, atol=atol)
        if ind > -1:
            nroots.pop(n)
            droots.pop(ind)
            #numpoly, rn = polydiv(numpoly, poly(curn))
            #denpoly, rd = polydiv(denpoly, poly(curn))
        else:
            n += 1
    numpoly = poly(nroots)
    denpoly = poly(droots)
    nvect = numpoly
    dvect = denpoly
    if prepend:
        nout, dout = prependzeros(nvect, dvect)
    else:
        nout = nvect
        dout = dvect
    return nout, dout


def polysubstitute(polyin, numsub, densub):
    """Substitute one polynomial into another to support Tustin and
    other c2d algorithms of a similar approach.  The idea is to make
    it easy to substitute

        a  z-1
    s = - -----
        T  z+1

    or other forms involving ratios of polynomials for s in a
    polynomial of s such as the numerator or denominator of a transfer
    function.

    For the tustin example above, numsub=a*(z-1) and densub=T*(z+1),
    where numsub and densub are scipy.poly1d instances.

    Note that this approach seems to have substantial floating point
    problems."""
    mys = TransferFunction(numsub, densub)
    out = 0.0
    no = polyin.order
    for n, coeff in enumerate(polyin.coeffs):
        curterm = coeff*mys**(no-n)
        out = out+curterm
    return out


def tustin_sub(polyin, T, a=2.0):
    numsub = a*poly1d([1.0,-1.0])
    densub = T*poly1d([1.0,1.0])
    out = polysubstitute(polyin, numsub, densub)
    out.myvar = 'z'
    return out
    

def create_swept_sine_input(maxt, dt, maxf, minf=0.0, deadtime=2.0):
    t = arange(0, maxt, dt)
    u = sweptsine(t, minf=minf, maxf=maxf)
    if deadtime:
        deadt = arange(0,deadtime, dt)
        zv = zeros_like(deadt)
        u = r_[zv, u, zv]
    return u

def create_swept_sine_t(maxt, dt, deadtime=2.0):
    t = arange(0, maxt, dt)
    if deadtime:
        deadt = arange(0,deadtime, dt)
        t = t+max(deadt)+dt
        tpost = deadt+max(t)+dt
        return r_[deadt, t, tpost]
    else:
        return t

def ADC(vectin, bits=9, vmax=2.5, vmin=-2.5):
    """Simulate the sampling portion of an analog-to-digital
    conversion by outputing an integer number of counts associate with
    each voltage in vectin."""
    dv = (vmax-vmin)/2**bits
    vect2 = clip(vectin, vmin, vmax)
    counts = vect2/dv
    return counts.astype(int)


def CountsToFloat(counts, bits=9, vmax=2.5, vmin=-2.5):
    """Convert the integer output of ADC to a floating point number by
    mulitplying by dv."""
    dv = (vmax-vmin)/2**bits
    return dv*counts


def epslist(listin, eps=1.0e-12):
    """Make a copy of listin and then check each element of the copy
    to see if its absolute value is greater than eps.  Set to zero all
    elements in the copied list whose absolute values are less than
    eps.  Return the copied list."""
    listout = copy.deepcopy(listin)
    for i in range(len(listout)):
        if abs(listout[i])<eps:
            listout[i] = 0.0
    return listout


def _PlotMatrixvsF(freqvect,matin,linetype='',linewidth=None, semilogx=True, allsolid=False, axis=None):
    mykwargs={}
    usepylab = False
    if axis is None:
       import pylab
       axis = pylab.gca()
       usepylab = True
    if len(shape(matin))==1:
        myargs=[freqvect,matin]
        if linetype:
            myargs.append(linetype)
        else:
            mykwargs.update(_getlinetype(axis))
        if linewidth:
            mykwargs['linewidth']=linewidth
        if semilogx:
            curline,=axis.semilogx(*myargs,**mykwargs)
        else:
            curline,=axis.plot(*myargs,**mykwargs)
        mylines=[curline]
#        _inccount()
    else:
        mylines=[]
        for q in range(shape(matin)[1]):
            myargs=[freqvect,matin[:,q]]
            if linetype:
                myargs.append(linetype)
            else:
                mykwargs.update(_getlinetype(axis))
            if linewidth:
                mykwargs['linewidth']=linewidth
            if semilogx:
                curline,=axis.semilogx(*myargs,**mykwargs)
            else:
                curline,=axis.plot(*myargs,**mykwargs)
            mylines.append(curline)
#            _inccount()
    return mylines


def _PlotMag(freqvect, bodein, linetype='-', linewidth=0, axis=None):
    if callable(bodein.dBmag):
        myvect=bodein.dBmag()
    else:
        myvect=bodein.dBmag
    return _PlotMatrixvsF(freqvect, myvect, linetype=linetype, linewidth=linewidth, axis=axis)


def _PlotPhase(freqvect, bodein, linetype='-', linewidth=0, axis=None):
    return _PlotMatrixvsF(freqvect,bodein.phase,linetype=linetype,linewidth=linewidth, axis=axis)


def _k_poles(TF,poleloc):
    L = TF.num(poleloc)/TF.den(poleloc)
    k = 1.0/abs(L)
    poles = TF._RLFindRoots([k])
    poles = TF._RLSortRoots(poles)
    return k,poles

def _checkpoles(poleloc,pnew):
    evect = abs(poleloc-array(pnew))
    ind = evect.argmin()
    pout = pnew[ind]
    return pout


def _realizable(num, den):
    realizable = False
    if not isscalar(den):        
        if isscalar(num):
            realizable = True
        elif len(den) >= len(num):
            realizable = True
    return realizable


def shape_u(uvect, slope):
    u_shaped = zeros_like(uvect)
    u_shaped[0] = uvect[0]

    N = len(uvect)

    for n in range(1, N):
        diff = uvect[n] - u_shaped[n-1]
        if diff > slope:
            u_shaped[n] = u_shaped[n-1] + slope
        elif diff < -1*slope:
            u_shaped[n] = u_shaped[n-1] - slope
        else:
            u_shaped[n] = uvect[n]
    return u_shaped


class TransferFunction(signal.lti):
    def __setattr__(self, attr, val):
        realizable = False
        if hasattr(self, 'den') and hasattr(self, 'num'):
           realizable = _realizable(self.num, self.den)
        if realizable:
           signal.lti.__setattr__(self, attr, val)
        else:
           self.__dict__[attr] = val

          
    def __init__(self, num, den, dt=0.01, maxt=5.0, myvar='s'):
        """num and den are either scalar constants or lists that are
        passed to scipy.poly1d to create a list of coefficients."""
        #print('in TransferFunction.__init__, dt=%s' % dt)
        if _realizable(num, den):
            signal.lti.__init__(self, num, den)
        self.num = poly1d(num)
        self.den = poly1d(den) 
        self.dt = dt
        self.myvar = myvar
        self.maxt = maxt


    def __repr__(self, labelstr='controls.TransferFunction'):
        nstr=str(self.num)#.strip()
        dstr=str(self.den)#.strip()
        nstr=nstr.replace('x',self.myvar)
        dstr=dstr.replace('x',self.myvar)
        n=len(dstr)
        m=len(nstr)
        shift=(n-m)/2*' '
        nstr=nstr.replace('\n','\n'+shift)
        tempstr=labelstr+'\n'+shift+nstr+'\n'+'-'*n+'\n '+dstr
        return tempstr


    def __call__(self,s,optargs=()):
        return self.num(s)/self.den(s)


    def __add__(self,other):
        if hasattr(other,'num') and hasattr(other,'den'):
            if len(self.den.coeffs)==len(other.den.coeffs) and \
                   (self.den.coeffs==other.den.coeffs).all():
                return TransferFunction(self.num+other.num,self.den)
            else:
                return TransferFunction(self.num*other.den+other.num*self.den,self.den*other.den)
        elif isinstance(other, int) or isinstance(other, float):
            return TransferFunction(other*self.den+self.num,self.den)
        else:
            raise ValueError, 'do not know how to add TransferFunction and '+str(other) +' which is of type '+str(type(other))

    def __radd__(self,other):
        return self.__add__(other)


    def __mul__(self,other):
        if isinstance(other, Digital_P_Control):
           return self.__class__(other.kp*self.num, self.den)
        elif hasattr(other,'num') and hasattr(other,'den'):
            if myeq(self.num,other.den) and myeq(self.den,other.num):
                return 1
            elif myeq(self.num,other.den):
                return self.__class__(other.num,self.den)
            elif myeq(self.den,other.num):
                return self.__class__(self.num,other.den)
            else:
               gain = self.gain*other.gain
               new_num, new_den = polyfactor(self.num*other.num, \
                                             self.den*other.den)
               newtf = self.__class__(new_num*gain, new_den)
               return newtf
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(other*self.num,self.den)


    def __pow__(self, expon):
        """Basically, go self*self*self as many times as necessary.  I
        haven't thought about negative exponents.  I don't think this
        would be hard, you would just need to keep dividing by self
        until you got the right answer."""
        assert expon >= 0, 'TransferFunction.__pow__ does not yet support negative exponents.'
        out = 1.0
        for n in range(expon):
            out *= self
        return out


    def __rmul__(self,other):
        return self.__mul__(other)


    def __div__(self,other):
        if hasattr(other,'num') and hasattr(other,'den'):
            if myeq(self.den,other.den):
                return TransferFunction(self.num,other.num)
            else:
                return TransferFunction(self.num*other.den,self.den*other.num)
        elif isinstance(other, int) or isinstance(other, float):
            return TransferFunction(self.num,other*self.den)


    def __rdiv__(self, other):
        print('calling TransferFunction.__rdiv__')
        return self.__div__(other)


    def __truediv__(self,other):
        return self.__div__(other)


    def _get_set_dt(self, dt=None):
        if dt is not None:
            self.dt = float(dt)
        return self.dt


    def ToLatex(self, eps=1e-12, fmt='%0.5g', ds=True):
        mynum = self.num
        myden = self.den
        npart = PolyToLatex(mynum)
        dpart = PolyToLatex(myden)
        outstr = '\\frac{'+npart+'}{'+dpart+'}'
        if ds:
            outstr = '\\displaystyle '+outstr
        return outstr


    def RootLocus(self, kvect, fig=None, fignum=1, \
                  clear=True, xlim=None, ylim=None, plotstr='-'):
        """Calculate the root locus by finding the roots of 1+k*TF(s)
        where TF is self.num(s)/self.den(s) and each k is an element
        of kvect."""
        if fig is None:
            import pylab
            fig = pylab.figure(fignum)
        if clear:
            fig.clf()
        ax = fig.add_subplot(111)
        mymat = self._RLFindRoots(kvect)
        mymat = self._RLSortRoots(mymat)
        #plot open loop poles
        poles = array(self.den.r)
        ax.plot(real(poles), imag(poles), 'x')
        #plot open loop zeros
        zeros = array(self.num.r)
        if zeros.any():
            ax.plot(real(zeros), imag(zeros), 'o')
        for col in mymat.T:
            ax.plot(real(col), imag(col), plotstr)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        return mymat


    def _RLFindRoots(self, kvect):
        """Find the roots for the root locus."""
        roots = []
        for k in kvect:
            curpoly = self.den+k*self.num
            curroots = curpoly.r
            curroots.sort()
            roots.append(curroots)
        mymat = row_stack(roots)
        return mymat


    def _RLSortRoots(self, mymat):
        """Sort the roots from self._RLFindRoots, so that the root
        locus doesn't show weird pseudo-branches as roots jump from
        one branch to another."""
        sorted = zeros_like(mymat)
        for n, row in enumerate(mymat):
            if n==0:
                sorted[n,:] = row
            else:
                #sort the current row by finding the element with the
                #smallest absolute distance to each root in the
                #previous row
                available = range(len(prevrow))
                for elem in row:
                    evect = elem-prevrow[available]
                    ind1 = abs(evect).argmin()
                    ind = available.pop(ind1)
                    sorted[n,ind] = elem
            prevrow = sorted[n,:]
        return sorted
        

    def opt(self, kguess):
        pnew = self._RLFindRoots(kguess)
        pnew = self._RLSortRoots(pnew)[0]
        if len(pnew)>1:
           pnew = _checkpoles(self.poleloc,pnew)
           e = abs(pnew-self.poleloc)**2
        return sum(e)


    def rlocfind(self, poleloc):
        self.poleloc = poleloc
        kinit,pinit = _k_poles(self,poleloc)
        k = optimize.fmin(self.opt,[kinit])[0]
        poles = self._RLFindRoots([k])
        poles = self._RLSortRoots(poles)
        return k, poles


    def PlotTimeResp(self, u, t, fig, clear=True, label='model', mysub=111):
        ax = fig.add_subplot(mysub)
        if clear:
            ax.cla()
        try:
            y = self.lsim(u, t)
        except:
            y = self.lsim2(u, t)
        ax.plot(t, y, label=label)
        return ax


##     def BodePlot(self, f, fig, clear=False):
##         mtf = self.FreqResp(
##         ax1 = fig.axes[0]
##         ax1.semilogx(modelf,20*log10(abs(mtf)))
##         mphase = angle(mtf, deg=1)
##         ax2 = fig.axes[1]
##         ax2.semilogx(modelf, mphase)

        
    def SimpleFactor(self):
        mynum=self.num
        myden=self.den
        dsf=myden[myden.order]
        nsf=mynum[mynum.order]
        sden=myden/dsf
        snum=mynum/nsf
        poles=sden.r
        residues=zeros(shape(sden.r),'D')
        factors=[]
        for x,p in enumerate(poles):
            polearray=poles.copy()
            polelist=polearray.tolist()
            mypole=polelist.pop(x)
            tempden=1.0
            for cp in polelist:
                tempden=tempden*(poly1d([1,-cp]))
            tempTF=TransferFunction(snum,tempden)
            curres=tempTF(mypole)
            residues[x]=curres
            curTF=TransferFunction(curres,poly1d([1,-mypole]))
            factors.append(curTF)
        return factors,nsf,dsf

    def factor_constant(self, const):
        """Divide numerator and denominator coefficients by const"""
        self.num = self.num/const
        self.den = self.den/const

    def lsim(self, u, t, interp=0, returnall=False, X0=None):
        """Find the response of the TransferFunction to the input u
        with time vector t.  Uses signal.lsim.

        return y the response of the system."""
        if returnall:#most users will just want the system output y,
                     #but some will need the (t, y, x) tuple that
                     #signal.lsim returns
            return signal.lsim(self, u, t, interp=interp, X0=X0)
        else:
            return signal.lsim(self, u, t, interp=interp, X0=X0)[1]

    def lsim2(self, u, t, returnall=False, X0=None):
        #tempsys=signal.lti(self.num,self.den)
        if returnall:
            return signal.lsim2(self, u, t, X0=X0)
        else:
            return signal.lsim2(self, u, t, X0=X0)[1]


    def residue(self, tol=1e-3, verbose=0):
        """from scipy.signal.residue:

        Compute residues/partial-fraction expansion of b(s) / a(s).

        If M = len(b) and N = len(a)

                b(s)     b[0] s**(M-1) + b[1] s**(M-2) + ... + b[M-1]
        H(s) = ------ = ----------------------------------------------
                a(s)     a[0] s**(N-1) + a[1] s**(N-2) + ... + a[N-1]

                 r[0]       r[1]             r[-1]
             = -------- + -------- + ... + --------- + k(s)
               (s-p[0])   (s-p[1])         (s-p[-1])

        If there are any repeated roots (closer than tol), then the
        partial fraction expansion has terms like

            r[i]      r[i+1]              r[i+n-1]
          -------- + ----------- + ... + -----------
          (s-p[i])  (s-p[i])**2          (s-p[i])**n

          returns r, p, k
          """
        r,p,k = signal.residue(self.num, self.den, tol=tol)
        if verbose>0:
            print('r='+str(r))
            print('')
            print('p='+str(p))
            print('')
            print('k='+str(k))

        return r, p, k


    def PartFrac(self, eps=1.0e-12):
        """Compute the partial fraction expansion based on the residue
        command.  In the final polynomials, coefficients whose
        absolute values are less than eps are set to zero."""
        r,p,k = self.residue()

        rlist = r.tolist()
        plist = p.tolist()

        N = len(rlist)

        tflist = []
        eps = 1e-12

        while N > 0:
            curr = rlist.pop(0)
            curp = plist.pop(0)
            if abs(curp.imag) < eps:
                #This is a purely real pole.  The portion of the partial
                #fraction expansion corresponding to this pole is curr/(s-curp)
                curtf = TransferFunction(curr,[1,-curp])
            else:
                #this is a complex pole and we need to find its conjugate and
                #handle them together
                cind = plist.index(curp.conjugate())
                rconj = rlist.pop(cind)
                pconj = plist.pop(cind)
                p1 = poly1d([1,-curp])
                p2 = poly1d([1,-pconj])
                #num = curr*p2+rconj*p1
                Nr = curr.real
                Ni = curr.imag
                Pr = curp.real
                Pi = curp.imag
                numlist = [2.0*Nr,-2.0*(Nr*Pr+Ni*Pi)]
                numlist = epslist(numlist, eps)
                num = poly1d(numlist)
                denlist = [1, -2.0*Pr,Pr**2+Pi**2]
                denlist = epslist(denlist, eps)
                den = poly1d(denlist)
                curtf = TransferFunction(num,den)
            tflist.append(curtf)
            N = len(rlist)
        return tflist
    

    def FreqResp(self, f, fignum=1, fig=None, clear=True, \
                 grid=True, legend=None, legloc=1, legsub=1, **kwargs):
        """Compute the frequency response of the transfer function
        using the frequency vector f, returning a complex vector.

        The frequency response (Bode plot) will be plotted on
        figure(fignum) unless fignum=None.

        legend should be a list of legend entries if a legend is
        desired.  If legend is not None, the legend will be placed on
        the top half of the plot (magnitude portion) if legsub=1, or
        on the bottom half with legsub=2.  legloc follows the same
        rules as the pylab legend command (1 is top right and goes
        counter-clockwise from there.)"""
        testvect=real(f)==0
        if testvect.all():
           s=f#then you really sent me s and not f
        else:
           s=2.0j*pi*f
        self.comp = self.num(s)/self.den(s)
        self.dBmag = 20*log10(abs(self.comp))
        self.phase = angle(self.comp,1)
        
        if fig is None:
            if fignum is not None:
                import pylab
                fig = pylab.figure(fignum)
            
        if fig is not None:
            if clear:
                fig.clf()

        if fig is not None:
            myargs=['linetype','colors','linewidth']
            subkwargs={}
            for key in myargs:
                if kwargs.has_key(key):
                    subkwargs[key]=kwargs[key]
            if clear:
                fig.clf()
            ax1 = fig.add_subplot(2,1,1)
            #if clear:
            #    ax1.cla()
            myind=ax1._get_lines.count
            mylines=_PlotMag(f, self, axis=ax1, **subkwargs)
            ax1.set_ylabel('Mag. Ratio (dB)')
            ax1.xaxis.set_major_formatter(MyFormatter())
            if grid:
               ax1.grid(1)
            if legend is not None and legsub==1:
               ax1.legend(legend, legloc)
            ax2 = fig.add_subplot(2,1,2, sharex=ax1)
            #if clear:
            #    ax2.cla()
            mylines=_PlotPhase(f, self, axis=ax2, **subkwargs)
            ax2.set_ylabel('Phase (deg.)')
            ax2.set_xlabel('Freq. (Hz)')
            ax2.xaxis.set_major_formatter(MyFormatter())
            if grid:
               ax2.grid(1)
            if legend is not None and legsub==2:
               ax2.legend(legend, legloc)
        return self.comp


    def CrossoverFreq(self, f):
       if not hasattr(self, 'dBmag'):
          self.FreqResp(f, fignum=None)
       t1 = squeeze(self.dBmag > 0.0)
       t2 = r_[t1[1:],t1[0]]
       t3 = (t1 & -t2)
       myinds = where(t3)[0]
       if not myinds.any():
          return None, []
       maxind = max(myinds)
       return f[maxind], maxind


    def PhaseMargin(self,f):
       fc,ind=self.CrossoverFreq(f)
       if not fc:
          return 180.0
       return 180.0+squeeze(self.phase[ind])


    def create_tvect(self, dt=None, maxt=None):
        if dt is None:
            dt = self.dt
        else:
            self.dt = dt
        assert dt is not None, "You must either pass in a dt or call create_tvect on an instance with a self.dt already defined."
        if maxt is None:
            if hasattr(self,'maxt'):
                maxt = self.maxt
            else:
                maxt = 100*dt
        else:
            self.maxt = maxt
        tvect = arange(0,maxt+dt/2.0,dt)
        self.t = tvect
        return tvect


    def create_impulse(self, dt=None, maxt=None, imp_time=0.5):
        """Create the input impulse vector to be used in least squares
        curve fitting of the c2d function."""
        if dt is None:
        	dt = self.dt
        indon = int(imp_time/dt)
        tvect = self.create_tvect(dt=dt, maxt=maxt)
        imp = zeros_like(tvect)
        imp[indon] = 1.0
        return imp
    

    def create_step_input(self, dt=None, maxt=None, indon=5):
        """Create the input impulse vector to be used in least squares
        curve fitting of the c2d function."""
        tvect = self.create_tvect(dt=dt, maxt=maxt)
        mystep = zeros_like(tvect)
        mystep[indon:] = 1.0
        return mystep


    def step_response(self, t=None, dt=None, maxt=None, \
                      step_time=None, fignum=1, clear=True, \
                      plotu=False, amp=1.0, interp=0, fig=None, \
                      fmts=['-','-'], legloc=0, returnall=0, \
                      legend=None, **kwargs):
        """Find the response of the system to a step input.  If t is
        not given, then the time vector will go from 0 to maxt in
        steps of dt i.e. t=arange(0,maxt,dt).  If dt and maxt are not
        given, the parameters from the TransferFunction instance will
        be used.

        step_time is the time when the step input turns on.  If not
        given, it will default to 0.

        If clear is True, the figure will be cleared first.
        clear=False could be used to overlay the step responses of
        multiple TransferFunction's.

        plotu=True means that the step input will also be shown on the
        graph.

        amp is the amplitude of the step input.

        return y unless returnall is set then return y, t, u

        where y is the response of the transfer function, t is the
        time vector, and u is the step input vector."""
        if t is not None:
           tvect = t
        else:
           tvect = self.create_tvect(dt=dt, maxt=maxt)
        u = zeros_like(tvect)
        if dt is None:
            dt = self.dt
        if step_time is None:
            step_time = 0.0
            #step_time = 0.1*tvect.max()
        if kwargs.has_key('indon'):
            indon = kwargs['indon']
        else:
            indon = int(step_time/dt)
        u[indon:] = amp
        try:
            ystep = self.lsim(u, tvect, interp=interp)#[1]#the outputs of lsim are (t, y,x)
        except:
            ystep = self.lsim2(u, tvect)#[1]

        if fig is None:
            if fignum is not None:
                import pylab
                fig = pylab.figure(fignum)
            
        if fig is not None:
            if clear:
                fig.clf()
            ax = fig.add_subplot(111)
            if plotu:
                leglist =['Input','Output'] 
                ax.plot(tvect, u, fmts[0], linestyle='steps', **kwargs)#assume step input wants 'steps' linestyle
                ofmt = fmts[1]
            else:
                ofmt = fmts[0]
            ax.plot(tvect, ystep, ofmt, **kwargs)
            ax.set_ylabel('Step Response')
            ax.set_xlabel('Time (sec)')
            if legend is not None:
               ax.legend(legend, loc=legloc)
            elif plotu:
                ax.legend(leglist, loc=legloc)
            #return ystep, ax
        #else:
            #return ystep
        if returnall:
           return ystep, tvect, u
        else:
           return ystep



    def impulse_response(self, dt=None, maxt=None, fignum=1, \
                         clear=True, amp=1.0, fig=None, \
                         fmt='-', **kwargs):
        """Find the impulse response of the system using
        scipy.signal.impulse.

        The time vector will go from 0 to maxt in steps of dt
        i.e. t=arange(0,maxt,dt).  If dt and maxt are not given, the
        parameters from the TransferFunction instance will be used.

        If clear is True, the figure will be cleared first.
        clear=False could be used to overlay the impulse responses of
        multiple TransferFunction's.

        amp is the amplitude of the impulse input.

        return y, t

        where y is the impulse response of the transfer function and t
        is the time vector."""

        tvect = self.create_tvect(dt=dt, maxt=maxt)
        temptf = amp*self
        tout, yout = temptf.impulse(T=tvect)

        if fig is None:
            if fignum is not None:
                import pylab
                fig = pylab.figure(fignum)
            
        if fig is not None:
            if clear:
                fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(tvect, yout, fmt, **kwargs)
            ax.set_ylabel('Impulse Response')
            ax.set_xlabel('Time (sec)')

        return yout, tout

        
    def swept_sine_response(self, maxf, minf=0.0, dt=None, maxt=None, deadtime=2.0, interp=0):
        u = create_swept_sine_input(maxt, dt, maxf, minf=minf, deadtime=deadtime)
        t = create_swept_sine_t(maxt, dt, deadtime=deadtime)
        ysweep = self.lsim(u, t, interp=interp)
        return t, u, ysweep

    
    def _c2d_sub(self, numsub, densub, scale):
        """This method performs substitutions for continuous to
        digital conversions using the form:

                    numsub
        s = scale* --------
                    densub

        where scale is a floating point number and numsub and densub
        are poly1d instances.

        For example, scale = 2.0/T, numsub = poly1d([1,-1]), and
        densub = poly1d([1,1]) for a Tustin c2d transformation."""
        m = self.num.order
        n = self.den.order
        mynum = 0.0
        for p, coeff in enumerate(self.num.coeffs):
            mynum += poly1d(coeff*(scale**(m-p))*((numsub**(m-p))*(densub**(n-(m-p)))))
        myden = 0.0
        for p, coeff in enumerate(self.den.coeffs):
            myden += poly1d(coeff*(scale**(n-p))*((numsub**(n-p))*(densub**(n-(n-p)))))
        return mynum.coeffs, myden.coeffs

        
    def c2d_tustin(self, dt=None, a=2.0):
        """Convert a continuous time transfer function into a digital
        one by substituting
        
            a  z-1
        s = - -----
            T  z+1

        into the compensator, where a is typically 2.0"""
        #print('in TransferFunction.c2d_tustin, dt=%s' % dt)
        dt = self._get_set_dt(dt)
        #print('in TransferFunction.c2d_tustin after _get_set_dt, dt=%s' % dt)
        scale = a/dt
        numsub = poly1d([1.0,-1.0])
        densub = poly1d([1.0,1.0])
        mynum, myden = self._c2d_sub(numsub, densub, scale)
        mynum = mynum/myden[0]
        myden = myden/myden[0]
        return mynum, myden
        
        

    def c2d(self, dt=None, maxt=None, method='zoh', step_time=0.5, a=2.0):
        """Find a numeric approximation of the discrete transfer
        function of the system.

        The general approach is to find the response of the system
        using lsim and fit a discrete transfer function to that
        response as a least squares problem.
        
        dt is the time between discrete time intervals (i.e. the
        sample time).

        maxt is the length of time for which to calculate the system
        respnose.  An attempt is made to guess an appropriate stopping
        time if maxt is None.  For now, this defaults to 100*dt,
        assuming that dt is appropriate for the system poles.

        method is a string describing the c2d conversion algorithm.
        method = 'zoh refers to a zero-order hold for a sampled-data
        system and follows the approach outlined by Dorsey in section
        14.17 of
        "Continuous and Discrete Control Systems" summarized on page
        472 of the 2002 edition.

        Other supported options for method include 'tustin'

        indon gives the index of when the step input should switch on
        for zoh or when the impulse should happen otherwise.  There
        should probably be enough zero entries before the input occurs
        to accomidate the order of the discrete transfer function.

        a is used only if method = 'tustin' and it is substituted in the form

             a  z-1
         s = - -----
             T  z+1

        a is almost always equal to 2.
        """
        if method.lower() == 'zoh':
            ystep = self.step_response(dt=dt, maxt=maxt, step_time=step_time)[0]
            myimp = self.create_impulse(dt=dt, maxt=maxt, imp_time=step_time)
            #Pdb().set_trace()
            print('You called c2d with "zoh".  This is most likely bad.')
            nz, dz = fit_discrete_response(ystep, myimp, self.den.order, self.den.order+1)#we want the numerator order to be one less than the denominator order - the denominator order +1 is the order of the denominator during a step response
            #multiply by (1-z^-1)
            nz2 = r_[nz, [0.0]]
            nzs = r_[[0.0],nz]
            nz3 = nz2 - nzs
            nzout, dzout = polyfactor(nz3, dz)
            return nzout, dzout
            #return nz3, dz
        elif method.lower() == 'tustin':
            #The basic approach for tustin is to create a transfer
            #function that represents s mapped into z and then
            #substitute this s(z)=a/T*(z-1)/(z+1) into the continuous
            #transfer function
            return self.c2d_tustin(dt=dt, a=a)
        else:
            raise ValueError, 'c2d method not understood:'+str(method)



    def DigitalSim(self, u, method='zoh', bits=9, vmin=-2.5, vmax=2.5, dt=None, maxt=None, digitize=True):
        """Simulate the digital reponse of the transfer to input u.  u
        is assumed to be an input signal that has been sampled with
        frequency 1/dt.  u is further assumed to be a floating point
        number with precision much higher than bits.  u will be
        digitized over the range [min, max], which is broken up into
        2**bits number of bins.

        The A and B vectors from c2d conversion will be found using
        method, dt, and maxt.  Note that maxt is only used for
        method='zoh'.

        Once A and B have been found, the digital reponse of the
        system to the digitized input u will be found."""
        B, A = self.c2d(dt=dt, maxt=maxt, method=method)
        assert A[0]==1.0, "A[0]!=1 in c2d result, A="+str(A)
        uvect = zeros(len(B), dtype='d')
        yvect = zeros(len(A)-1, dtype='d')
        if digitize:
            udig = ADC(u, bits, vmax=vmax, vmin=vmin)
            dv = (vmax-vmin)/(2**bits-1)
        else:
            udig = u
            dv = 1.0
        Ydig = zeros(len(u), dtype='d')
        for n, u0 in enumerate(udig):
            uvect = shift(uvect, u0)
            curY = dot(uvect,B)
            negpart = dot(yvect,A[1:])
            curY -= negpart
            if digitize:
                curY = int(curY)
            Ydig[n] = curY
            yvect = shift(yvect, curY)
        return Ydig*dv



class Input(TransferFunction):
    def __repr__(self):
        return TransferFunction.__repr__(self, labelstr='controls.Input')


class Compensator(TransferFunction):
   def __init__(self, num, den, *args, **kwargs):
      #print('in Compensator.__init__')
      #Pdb().set_trace()
      TransferFunction.__init__(self, num, den, *args, **kwargs)
      
      
   def c2d(self, dt=None, a=2.0):
      """Compensators should use Tustin for c2d conversion.  This
      method is just and alias for TransferFunction.c2d_tustin"""
      #print('in Compensators.c2d, dt=%s' % dt)
      #Pdb().set_trace()
      return TransferFunction.c2d_tustin(self, dt=dt, a=a)

   def __repr__(self):
        return TransferFunction.__repr__(self, labelstr='controls.Compensator')



class Digital_Compensator(object):
   def __init__(self, num, den, input_vect=None, output_vect=None):
      self.num = num
      self.den = den
      self.input = input_vect
      self.output = output_vect
      self.Nnum = len(self.num)
      self.Nden = len(self.den)
      

   def calc_out(self, i):
      out = 0.0
      for n, bn in enumerate(self.num):
         out += self.input[i-n]*bn

      for n in range(1, self.Nden):
         out -= self.output[i-n]*self.den[n]
      out = out/self.den[0]
      return out


class Digital_PI(object):
   def __init__(self, kp, ki, input_vect=None, output_vect=None):
      self.kp = kp
      self.ki = ki
      self.input = input_vect
      self.output = output_vect
      self.esum = 0.0


   def prep(self):
      self.esum = zeros_like(self.input)


   def calc_out(self, i):
      self.esum[i] = self.esum[i-1]+self.input[i]
      out = self.input[i]*self.kp+self.esum[i]*self.ki
      return out
      

class Digital_P_Control(Digital_Compensator):
   def __init__(self, kp, input_vect=None, output_vect=None):
      self.kp = kp
      self.input = input_vect
      self.output = output_vect
      self.num = poly1d([kp])
      self.den = poly1d([1])
      self.gain = 1

   def calc_out(self, i):
      self.output[i] = self.kp*self.input[i]
      return self.output[i]
   

def dig_comp_from_c_comp(c_comp, dt):
   """Convert a continuous compensator into a digital one using Tustin
   and sampling time dt."""
   b, a = c_comp.c2d_tustin(dt=dt)
   return Digital_Compensator(b, a)


class FirstOrderCompensator(Compensator):
   def __init__(self, K, z, p, dt=0.004):
      """Create a first order compensator whose transfer function is

               K*(s+z)
      D(s) = -----------
                (s+p)    """
      Compensator.__init__(self, K*poly1d([1,z]), [1,p])
      

   def __repr__(self):
        return TransferFunction.__repr__(self, labelstr='controls.FirstOrderCompensator')


   def ToPSoC(self, dt=0.004):
      b, a = self.c2d(dt=dt)
      outstr = 'v = %f*e%+f*ep%+f*vp;'%(b[0],b[1],-a[1])
      print('PSoC str:')
      print(outstr)
      return outstr


def sat(vin, vmax=2.0):
    if vin > vmax:
        return vmax
    elif vin < -1*vmax:
        return -1*vmax
    else:
        return vin



class Closed_Loop_System_with_Sat(object):
   def __init__(self, plant_tf, Kp, sat):
      self.plant_tf = plant_tf
      self.Kp = Kp
      self.sat = sat


   def lsim(self, u, t, X0=None, include_sat=True, \
            returnall=0, lsim2=0, verbosity=0):
      dt = t[1]-t[0]
      if X0 is None:
         X0 = zeros((2,len(self.plant_tf.den.coeffs)-1))
      N = len(t)
      y = zeros(N)
      v = zeros(N)
      x_n = X0
      for n in range(1,N):
          t_n = t[n]
          if verbosity > 0:
             print('t_n='+str(t_n))
          e = u[n]-y[n-1]
          v_n = self.Kp*e
          if include_sat:
              v_n = sat(v_n, vmax=self.sat)
          #simulate for one dt using ZOH
          if lsim2:
             t_nn, y_n, x_n = self.plant_tf.lsim2([v_n,v_n], [t_n, t_n+dt], X0=x_n[-1], returnall=1)
          else:
             t_nn, y_n, x_n = self.plant_tf.lsim([v_n,v_n], [t_n, t_n+dt], X0=x_n[-1], returnall=1)
             
          y[n] = y_n[-1]
          v[n] = v_n
      self.y = y
      self.v = v
      self.u = u
      if returnall:
         return y, v
      else:
         return y


      
      
      
def step_input():
    return Input(1,[1,0])

    
def feedback(olsys,H=1):
    """Calculate the closed-loop transfer function

                 olsys
      cltf = --------------
              1+H*olsys

     where olsys is the transfer function of the open loop
     system (Gc*Gp) and H is the transfer function in the feedback
     loop (H=1 for unity feedback)."""
    clsys=olsys/(1.0+H*olsys)
    return clsys



def Usweep(ti,maxt,minf=0.0,maxf=10.0):
    """Return the current value (scalar) of a swept sine signal - must be used
    with list comprehension to generate a vector.

    ti - current time (scalar)
    minf - lowest frequency in the sweep
    maxf - highest frequency in the sweep
    maxt - T or the highest value in the time vector"""
    if ti<0.0:
        return 0.0
    else:
        curf=(maxf-minf)*ti/maxt+minf
        if ti<(maxt*0.95):
            return sin(2*pi*curf*ti)
        else:
            return 0.0


def sweptsine(t,minf=0.0, maxf=10.0):
    """Generate a sweptsine vector by calling Usweep for each ti in t."""
    T=max(t)-min(t)
    Us = [Usweep(ti,T,minf,maxf) for ti in t]
    return array(Us)


mytypes=['-','--',':','-.']
colors=['b','y','r','g','c','k']#['y','b','r','g','c','k']

def _getlinetype(ax=None):
    if ax is None:
       import pylab
       ax = pylab.gca()
    myind=ax._get_lines.count
    return {'color':colors[myind % len(colors)],'linestyle':mytypes[myind % len(mytypes)]}


def create_step_vector(t, step_time=0.0, amp=1.0):
   u = zeros_like(t)
   dt = t[1]-t[0]
   indon = int(step_time/dt)
   u[indon:] = amp
   return u


def rate_limiter(uin, du):
   uout = zeros_like(uin)
   N = len(uin)
   for n in range(1,N):
      curchange = uin[n]-uout[n-1]
      if curchange > du:
         uout[n] = uout[n-1]+du
      elif curchange < -du:
         uout[n] = uout[n-1]-du
      else:
         uout[n] = uin[n]
   return uout



   
