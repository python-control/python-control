# phaseplot.py - generate 2D phase portraits
#
# Author: Richard M. Murray
# Date: Fall 2002 (MATLAB version), based on a version by Kristi Morgansen
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
from matplotlib.mlab import frange, find
from exception import ControlNotImplemented
from scipy.integrate import odeint

def PhasePlot(odefun, xlimv, ylimv, scale=1, xinit=None, T=None, parms=(),
              verbose=True):
    """
    PhasePlot   Phase plot for 2D dynamical systems

    PhasePlot(F, X1range, X2range, scale) produces a quiver plot for
    the function F.  X1range and X2range have the form [min, max, num]
    and specify the axis limits for the plot, along with the number of
    subdivisions in each axis, used to produce an quiver plot for the
    vector field.  The vector field is scaled by a factor 'scale'
    (default = 1).

    The function F should be the same for as used for scipy.integrate.
    Namely, it should be a function of the form dxdt = F(x, t) that
    accepts a state x of dimension 2 and returns a derivative dxdt of
    dimension 2.

    PhasePlot(F, X1range, X2range, scale, Xinit) produces a phase
    plot for the function F, consisting of the quiver plot plus stream
    lines.  The streamlines start at the initial conditions listed in
    Xinit, which should be a matrix whose rows give the desired inital
    conditions for x1 and x2.  X1range or X2range is 'auto', the arrows
    are produced based on the stream lines.  If 'scale' is negative,
    dots are placed at the base of the arrows.  If 'scale' is zero, no
    dots are produced.
 					
    PhasePlot(F, X1range, X2range, scale, boxgrid(X1range2, X2range2))
    produces a phase plot with stream lines generated at the edges of
    the rectangle defined by X1range2, X2range2.  These ranges are in
    the same form as X1range, X2range.

    PhasePlot(F, X1range, X2range, scale, Xinit, T) produces a phase
    plot where the streamlines are simluted for time T (default = 50).

    PhasePlot(F, X1range, X2range, scale, Xinit, T, P1, P2, ...)
    passes additional parameters to the function F, in the same way as
    ODE45.

    Instead of drawing arrows on a grid, arrows can also be drawn on
    streamlines by usinge the X1range and X2range arguments as follows:

    X1range	X2range
    -------	-------
    'auto'	N	  Draw N arrows using equally space time points
    'logtime'   [N, lam]  Draw N arrows using exponential time constant lam
    'timepts'	[t1, t2, ...]  Draw arrows at the list times
    """

    #
    # Parameters defining the plot
    #
    # The constants below define the parameters that control how the
    # plot looks.  These can be modified to customize the look of the
    # phase plot.
    #

    #! TODO: convert to keywords
    #! TODO: eliminate old parameters that aren't used
    # PP color = ['m', 'c', 'r', 'g', 'b', 'k', 'y'];
    PP_stream_color = ('b');		# color array for streamlines
    PP_stream_linewidth = 1;		# line width for stream lines

    PP_arrow_linewidth = 1;		# line width for arrows (quiver)
    PP_arrow_markersize = 10;		# size of arrow base marker

    #
    # Figure out ranges for phase plot (argument processing)
    #
    auto = 0; logtime = 0; timepts = 0; Narrows = 0;
    if (isinstance(xlimv, str) and xlimv == 'auto'):
        auto = 1;
        Narrows = ylimv;
        if (verbose):
            print 'Using auto arrows\n';

    elif (isinstance(xlimv, str) and xlimv == 'logtime'):
        logtime = 1;
        Narrows = ylimv[0];
        timefactor = ylimv[1];
        if (verbose):
            print 'Using logtime arrows\n';

    elif (isinstance(xlimv, str) and xlimv == 'timepts'):
        timepts = 1;
        Narrows = len(ylimv);

    else:
        # Figure out the set of points for the quiver plot
        (x1, x2) = np.meshgrid(
            frange(xlimv[0], xlimv[1], float(xlimv[1]-xlimv[0])/xlimv[2]),
            frange(ylimv[0], ylimv[1], float(ylimv[1]-ylimv[0])/ylimv[2]));

    if ((not auto) and (not logtime) and (not timepts)):
        # Now calculate the vector field at those points
        (nr,nc) = x1.shape;
        dx = np.empty((nr, nc, 2))
        for i in range(nr):
            for j in range(nc):
                dx[i, j, :] = np.squeeze(odefun((x1[i,j], x2[i,j]), 0, *parms))

        # Plot the quiver plot
        #! TODO: figure out arguments to make arrows show up correctly
        if (scale == None):
            mpl.quiver(x1, x2, dx[:,:,1], dx[:,:,2], angles='xy')
        elif (scale != 0):
            #! TODO: optimize parameters for arrows
            #! TODO: figure out arguments to make arrows show up correctly
            xy = mpl.quiver(x1, x2, dx[:,:,0]*np.abs(scale), 
                            dx[:,:,1]*np.abs(scale), angles='xy')
            # set(xy, 'LineWidth', PP_arrow_linewidth, 'Color', 'b');

        #! TODO: Tweak the shape of the plot 
        # a=gca; set(a,'DataAspectRatio',[1,1,1]);
        # set(a,'XLim',xlimv(1:2)); set(a,'YLim',ylimv(1:2));
        mpl.xlabel('x1'); mpl.ylabel('x2');

    # See if we should also generate the streamlines
    if (xinit == None or len(xinit) == 0):
        return

    # Convert initial conditions to a numpy array
    xinit = np.array(xinit);
    (nr, nc) = np.shape(xinit);

    # Generate some empty matrices to keep arrow information
    x1 = np.empty((nr, Narrows)); x2 = np.empty((nr, Narrows));
    dx = np.empty((nr, Narrows, 2))
  
    # See if we were passed a simulation time
    if (T == None):
        T = 50

    # Parse the time we were passed
    TSPAN = T;
    if (isinstance(T, (int, float))):
        TSPAN = np.linspace(0, T, 100);

    # Figure out the limits for the plot
    if (scale == None):
        # Assume that the current axis are set as we want them
        alim = mpl.axis();
        xmin = alim[0]; xmax = alim[1]; 
        ymin = alim[2]; ymax = alim[3];
    else:
        # Use the maximum extent of all trajectories
        xmin = np.min(xinit[:,0]); xmax = np.max(xinit[:,0]);
        ymin = np.min(xinit[:,1]); ymax = np.max(xinit[:,1]);

    # Generate the streamlines for each initial condition
    for i in range(nr):
        state = odeint(odefun, xinit[i], TSPAN, args=parms);
        time = TSPAN
        mpl.hold(True);
        mpl.plot(state[:,0], state[:,1])
        #! TODO: add back in colors for stream lines
        # PP_stream_color(np.mod(i-1, len(PP_stream_color))+1));
        # set(h[i], 'LineWidth', PP_stream_linewidth);

        # Plot arrows if quiver parameters were 'auto'
        if (auto or logtime or timepts):
            # Compute the locations of the arrows
            #! TODO: check this logic to make sure it works in python
            for j in range(Narrows):
      
                # Figure out starting index; headless arrows start at 0
                k = -1 if scale == None else 0;
      
                # Figure out what time index to use for the next point
                if (auto):
                    # Use a linear scaling based on ODE time vector
                    tind = np.floor((len(time)/Narrows) * (j-k)) + k;
                elif (logtime):
                    # Use an exponential time vector
                    # MATLAB: tind = find(time < (j-k) / lambda, 1, 'last');
                    tarr = find(time < (j-k) / timefactor);
                    tind = tarr[-1] if len(tarr) else 0;
                elif (timepts):
                    # Use specified time points
                    # MATLAB: tind = find(time < ylimv[j], 1, 'last');
                    tarr = find(time < ylimv[j]);
                    tind = tarr[-1] if len(tarr) else 0;

                # For tailless arrows, skip the first point
                if (tind == 0 and scale == None):
                    continue;
      
                # Figure out the arrow at this point on the curve
                x1[i,j] = state[tind, 0];
                x2[i,j] = state[tind, 1];

                # Skip arrows outside of initial condition box
                if (scale != None or 
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
    if (scale == None and Narrows > 0):
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
        bp = mpl.plot(x1, x2, 'b.');		# add dots at base
        # set(bp, 'MarkerSize', PP_arrow_markersize);

    return;
