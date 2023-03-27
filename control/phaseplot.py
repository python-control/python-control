#! TODO: add module docstring
# phaseplot.py - generate 2D phase portraits
#
# Author: Richard M. Murray
# Date: 24 July 2011, converted from MATLAB version (2002); based on
# a version by Kristi Morgansen
#
# Copyright (c) 2011 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   3. The name of the author may not be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import matplotlib.pyplot as mpl

from scipy.integrate import odeint
from .exception import ControlNotImplemented

__all__ = ['phase_plot', 'box_grid']


def _find(condition):
    """Returns indices where ravel(a) is true.
    Private implementation of deprecated matplotlib.mlab.find
    """
    return np.nonzero(np.ravel(condition))[0]


def phase_plot(odefun, X=None, Y=None, scale=1, X0=None, T=None,
              lingrid=None, lintime=None, logtime=None, timepts=None,
              parms=(), verbose=True):
    """Phase plot for 2D dynamical systems

    Produces a vector field or stream line plot for a planar system.

    Call signatures:
      phase_plot(func, X, Y, ...) - display vector field on meshgrid
      phase_plot(func, X, Y, scale, ...) - scale arrows
      phase_plot(func. X0=(...), T=Tmax, ...) - display stream lines
      phase_plot(func, X, Y, X0=[...], T=Tmax, ...) - plot both
      phase_plot(func, X0=[...], T=Tmax, lingrid=N, ...) - plot both
      phase_plot(func, X0=[...], lintime=N, ...) - stream lines with arrows

    Parameters
    ----------
    func : callable(x, t, ...)
        Computes the time derivative of y (compatible with odeint).
        The function should be the same for as used for
        :mod:`scipy.integrate`.  Namely, it should be a function of the form
        dxdt = F(x, t) that accepts a state x of dimension 2 and
        returns a derivative dx/dt of dimension 2.

    X, Y: 3-element sequences, optional, as [start, stop, npts]
        Two 3-element sequences specifying x and y coordinates of a
        grid.  These arguments are passed to linspace and meshgrid to
        generate the points at which the vector field is plotted.  If
        absent (or None), the vector field is not plotted.

    scale: float, optional
        Scale size of arrows; default = 1

    X0: ndarray of initial conditions, optional
        List of initial conditions from which streamlines are plotted.
        Each initial condition should be a pair of numbers.

    T: array-like or number, optional
        Length of time to run simulations that generate streamlines.
        If a single number, the same simulation time is used for all
        initial conditions.  Otherwise, should be a list of length
        len(X0) that gives the simulation time for each initial
        condition.  Default value = 50.

    lingrid : integer or 2-tuple of integers, optional
        Argument is either N or (N, M).  If X0 is given and X, Y are missing,
        a grid of arrows is produced using the limits of the initial
        conditions, with N grid points in each dimension or N grid points in x
        and M grid points in y.

    lintime : integer or tuple (integer, float), optional
        If a single integer N is given, draw N arrows using equally space time
        points.  If a tuple (N, lambda) is given, draw N arrows using
        exponential time constant lambda

    timepts : array-like, optional
        Draw arrows at the given list times [t1, t2, ...]

    parms: tuple, optional
        List of parameters to pass to vector field: `func(x, t, *parms)`

    See also
    --------
    box_grid : construct box-shaped grid of initial conditions

    """

    #
    # Figure out ranges for phase plot (argument processing)
    #
    #! TODO: need to add error checking to arguments
    #! TODO: think through proper action if multiple options are given
    #
    autoFlag = False; logtimeFlag = False; timeptsFlag = False; Narrows = 0;

    if lingrid is not None:
        autoFlag = True;
        Narrows = lingrid;
        if (verbose):
            print('Using auto arrows\n')

    elif logtime is not None:
        logtimeFlag = True;
        Narrows = logtime[0];
        timefactor = logtime[1];
        if (verbose):
            print('Using logtime arrows\n')

    elif timepts is not None:
        timeptsFlag = True;
        Narrows = len(timepts);

    # Figure out the set of points for the quiver plot
    #! TODO: Add sanity checks
    elif (X is not None and Y is not None):
        (x1, x2) = np.meshgrid(
            np.linspace(X[0], X[1], X[2]),
            np.linspace(Y[0], Y[1], Y[2]))
        Narrows = len(x1)

    else:
        # If we weren't given any grid points, don't plot arrows
        Narrows = 0;

    if ((not autoFlag) and (not logtimeFlag) and (not timeptsFlag)
        and (Narrows > 0)):
        # Now calculate the vector field at those points
        (nr,nc) = x1.shape;
        dx = np.empty((nr, nc, 2))
        for i in range(nr):
            for j in range(nc):
                dx[i, j, :] = np.squeeze(odefun((x1[i,j], x2[i,j]), 0, *parms))

        # Plot the quiver plot
        #! TODO: figure out arguments to make arrows show up correctly
        if scale is None:
            mpl.quiver(x1, x2, dx[:,:,1], dx[:,:,2], angles='xy')
        elif (scale != 0):
            #! TODO: optimize parameters for arrows
            #! TODO: figure out arguments to make arrows show up correctly
            xy = mpl.quiver(x1, x2, dx[:,:,0]*np.abs(scale),
                            dx[:,:,1]*np.abs(scale), angles='xy')
            # set(xy, 'LineWidth', PP_arrow_linewidth, 'Color', 'b');

        #! TODO: Tweak the shape of the plot
        # a=gca; set(a,'DataAspectRatio',[1,1,1]);
        # set(a,'XLim',X(1:2)); set(a,'YLim',Y(1:2));
        mpl.xlabel('x1'); mpl.ylabel('x2');

    # See if we should also generate the streamlines
    if X0 is None or len(X0) == 0:
        return

    # Convert initial conditions to a numpy array
    X0 = np.array(X0);
    (nr, nc) = np.shape(X0);

    # Generate some empty matrices to keep arrow information
    x1 = np.empty((nr, Narrows)); x2 = np.empty((nr, Narrows));
    dx = np.empty((nr, Narrows, 2))

    # See if we were passed a simulation time
    if T is None:
        T = 50

    # Parse the time we were passed
    TSPAN = T;
    if (isinstance(T, (int, float))):
        TSPAN = np.linspace(0, T, 100);

    # Figure out the limits for the plot
    if scale is None:
        # Assume that the current axis are set as we want them
        alim = mpl.axis();
        xmin = alim[0]; xmax = alim[1];
        ymin = alim[2]; ymax = alim[3];
    else:
        # Use the maximum extent of all trajectories
        xmin = np.min(X0[:,0]); xmax = np.max(X0[:,0]);
        ymin = np.min(X0[:,1]); ymax = np.max(X0[:,1]);

    # Generate the streamlines for each initial condition
    for i in range(nr):
        state = odeint(odefun, X0[i], TSPAN, args=parms);
        time = TSPAN

        mpl.plot(state[:,0], state[:,1])
        #! TODO: add back in colors for stream lines
        # PP_stream_color(np.mod(i-1, len(PP_stream_color))+1));
        # set(h[i], 'LineWidth', PP_stream_linewidth);

        # Plot arrows if quiver parameters were 'auto'
        if (autoFlag or logtimeFlag or timeptsFlag):
            # Compute the locations of the arrows
            #! TODO: check this logic to make sure it works in python
            for j in range(Narrows):

                # Figure out starting index; headless arrows start at 0
                k = -1 if scale is None else 0;

                # Figure out what time index to use for the next point
                if (autoFlag):
                    # Use a linear scaling based on ODE time vector
                    tind = np.floor((len(time)/Narrows) * (j-k)) + k;
                elif (logtimeFlag):
                    # Use an exponential time vector
                    # MATLAB: tind = find(time < (j-k) / lambda, 1, 'last');
                    tarr = _find(time < (j-k) / timefactor);
                    tind = tarr[-1] if len(tarr) else 0;
                elif (timeptsFlag):
                    # Use specified time points
                    # MATLAB: tind = find(time < Y[j], 1, 'last');
                    tarr = _find(time < timepts[j]);
                    tind = tarr[-1] if len(tarr) else 0;

                # For tailless arrows, skip the first point
                if tind == 0 and scale is None:
                    continue;

                # Figure out the arrow at this point on the curve
                x1[i,j] = state[tind, 0];
                x2[i,j] = state[tind, 1];

                # Skip arrows outside of initial condition box
                if (scale is not None or
                     (x1[i,j] <= xmax and x1[i,j] >= xmin and
                      x2[i,j] <= ymax and x2[i,j] >= ymin)):
                    v = odefun((x1[i,j], x2[i,j]), 0, *parms)
                    dx[i, j, 0] = v[0]; dx[i, j, 1] = v[1];
                else:
                    dx[i, j, 0] = 0; dx[i, j, 1] = 0;

    # Set the plot shape before plotting arrows to avoid warping
    # a=gca;
    # if (scale != None):
    #     set(a,'DataAspectRatio', [1,1,1]);
    # if (xmin != xmax and ymin != ymax):
    #     mpl.axis([xmin, xmax, ymin, ymax]);
    # set(a, 'Box', 'on');

    # Plot arrows on the streamlines
    if scale is None and Narrows > 0:
        # Use a tailless arrow
        #! TODO: figure out arguments to make arrows show up correctly
        mpl.quiver(x1, x2, dx[:,:,0], dx[:,:,1], angles='xy')
    elif (scale != 0 and Narrows > 0):
        #! TODO: figure out arguments to make arrows show up correctly
        xy = mpl.quiver(x1, x2, dx[:,:,0]*abs(scale), dx[:,:,1]*abs(scale),
                        angles='xy')
        # set(xy, 'LineWidth', PP_arrow_linewidth);
        # set(xy, 'AutoScale', 'off');
        # set(xy, 'AutoScaleFactor', 0);

    if (scale < 0):
        bp = mpl.plot(x1, x2, 'b.');        # add dots at base
        # set(bp, 'MarkerSize', PP_arrow_markersize);

    return;

# Utility function for generating initial conditions around a box
def box_grid(xlimp, ylimp):
    """box_grid   generate list of points on edge of box

    list = box_grid([xmin xmax xnum], [ymin ymax ynum]) generates a
    list of points that correspond to a uniform grid at the end of the
    box defined by the corners [xmin ymin] and [xmax ymax].
    """

    sx10 = np.linspace(xlimp[0], xlimp[1], xlimp[2])
    sy10 = np.linspace(ylimp[0], ylimp[1], ylimp[2])

    sx1 = np.hstack((0, sx10, 0*sy10+sx10[0], sx10, 0*sy10+sx10[-1]))
    sx2 = np.hstack((0, 0*sx10+sy10[0], sy10, 0*sx10+sy10[-1], sy10))

    return np.transpose( np.vstack((sx1, sx2)) )
