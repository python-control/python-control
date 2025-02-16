# phaseplot.py - generate 2D phase portraits
#
# Initial author: Richard M. Murray
# Creation date: 24 July 2011, converted from MATLAB version (2002);
# based on an original version by Kristi Morgansen

"""Generate 2D phase portraits.

This module contains functions for generating 2D phase plots. The base
function for creating phase plane portraits is `~control.phase_plane_plot`,
which generates a phase plane portrait for a 2 state I/O system (with no
inputs). Utility functions are available to customize the individual
elements of a phase plane portrait.

The docstring examples assume the following import commands::

  >>> import numpy as np
  >>> import control as ct
  >>> import control.phaseplot as pp

"""

import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from . import config
from .ctrlplot import ControlPlot, _add_arrows_to_line2D, _get_color, \
    _process_ax_keyword, _update_plot_title
from .exception import ControlArgument
from .nlsys import NonlinearIOSystem, find_operating_point, \
    input_output_response

__all__ = ['phase_plane_plot', 'phase_plot', 'box_grid']

# Default values for module parameter variables
_phaseplot_defaults = {
    'phaseplot.arrows': 2,                  # number of arrows around curve
    'phaseplot.arrow_size': 8,              # pixel size for arrows
    'phaseplot.arrow_style': None,          # set arrow style
    'phaseplot.separatrices_radius': 0.1    # initial radius for separatrices
}


def phase_plane_plot(
        sys, pointdata=None, timedata=None, gridtype=None, gridspec=None,
        plot_streamlines=None, plot_vectorfield=None, plot_streamplot=None,
        plot_equilpoints=True, plot_separatrices=True, ax=None,
        suppress_warnings=False, title=None, **kwargs
):
    """Plot phase plane diagram.

    This function plots phase plane data, including vector fields, stream
    lines, equilibrium points, and contour curves.
    If none of plot_streamlines, plot_vectorfield, or plot_streamplot are
    set, then plot_streamplot is used by default.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data. If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot or an array of shape (N, 2)
        giving points of at which to plot the vector field.
    timedata : int or list of int
        Time to simulate each streamline.  If a list is given, a different
        time can be used for each initial condition in `pointdata`.
    gridtype : str, optional
        The type of grid to use for generating initial conditions:
        'meshgrid' (default) generates a mesh of initial conditions within
        the specified boundaries, 'boxgrid' generates initial conditions
        along the edges of the boundary, 'circlegrid' generates a circle of
        initial conditions around each point in point data.
    gridspec : list, optional
        If the gridtype is 'meshgrid' and 'boxgrid', `gridspec` gives the
        size of the grid in the x and y axes on which to generate points.
        If gridtype is 'circlegrid', then `gridspec` is a 2-tuple
        specifying the radius and number of points around each point in the
        `pointdata` array.
    params : dict, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : matplotlib color spec, optional
        Plot all elements in the given color (use ``plot_<element>`` =
        {'color': c} to set the color in one element of the phase
        plot (equilpoints, separatrices, streamlines, etc).
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified and
        the current figure has a single axes, that axes is used.
        Otherwise, a new figure is created.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : array of list of `matplotlib.lines.Line2D`
        Array of list of `matplotlib.artist.Artist` objects:

            - lines[0] = list of Line2D objects (streamlines, separatrices).
            - lines[1] = Quiver object (vector field arrows).
            - lines[2] = list of Line2D objects (equilibrium points).
            - lines[3] = StreamplotSet object (lines with arrows).

    cplt.axes : 2D array of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.

    Other Parameters
    ----------------
    arrows : int
        Set the number of arrows to plot along the streamlines. The default
        value can be set in `config.defaults['phaseplot.arrows']`.
    arrow_size : float
        Set the size of arrows to plot along the streamlines.  The default
        value can be set in `config.defaults['phaseplot.arrow_size']`.
    arrow_style : matplotlib patch
        Set the style of arrows to plot along the streamlines.  The default
        value can be set in `config.defaults['phaseplot.arrow_style']`.
    dir : str, optional
        Direction to draw streamlines: 'forward' to flow forward in time
        from the reference points, 'reverse' to flow backward in time, or
        'both' to flow both forward and backward.  The amount of time to
        simulate in each direction is given by the `timedata` argument.
    plot_streamlines : bool or dict, optional
        If True then plot streamlines based on the pointdata and gridtype.
        If set to a dict, pass on the key-value pairs in the dict as
        keywords to `streamlines`.
    plot_vectorfield : bool or dict, optional
        If True then plot the vector field based on the pointdata and
        gridtype.  If set to a dict, pass on the key-value pairs in the
        dict as keywords to `phaseplot.vectorfield`.
    plot_streamplot : bool or dict, optional
        If True then use `matplotlib.axes.Axes.streamplot` function
        to plot the streamlines.  If set to a dict, pass on the key-value
        pairs in the dict as keywords to `phaseplot.streamplot`.
    plot_equilpoints : bool or dict, optional
        If True (default) then plot equilibrium points based in the phase
        plot boundary. If set to a dict, pass on the key-value pairs in the
        dict as keywords to `phaseplot.equilpoints`.
    plot_separatrices : bool or dict, optional
        If True (default) then plot separatrices starting from each
        equilibrium point.  If set to a dict, pass on the key-value pairs
        in the dict as keywords to `phaseplot.separatrices`.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    suppress_warnings : bool, optional
        If set to True, suppress warning messages in generating trajectories.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).

    Notes
    -----
    The default method for producing streamlines is determined based on which
    keywords are specified, with `plot_streamplot` serving as the generic
    default.  If any of the `arrows`, `arrow_size`, `arrow_style`, or `dir`
    keywords are used and neither `plot_streamlines` nor `plot_streamplot` is
    set, then `plot_streamlines` will be set to True.  If neither
    `plot_streamlines` nor `plot_vectorfield` set set to True, then
    `plot_streamplot` will be set to True.

    """
    # Check for legacy usage of plot_streamlines
    streamline_keywords = [
        'arrows', 'arrow_size', 'arrow_style', 'dir']
    if plot_streamlines is None:
        if any([kw in kwargs for kw in streamline_keywords]):
            warnings.warn(
                "detected streamline keywords; use plot_streamlines to set",
                FutureWarning)
            plot_streamlines = True
        if gridtype not in [None, 'meshgrid']:
            warnings.warn(
                "streamplots only support gridtype='meshgrid'; "
                "falling back to streamlines")
            plot_streamlines = True

    if plot_streamlines is None and plot_vectorfield is None \
       and plot_streamplot is None:
        plot_streamplot = True

    if plot_streamplot and not plot_streamlines and not plot_vectorfield:
        gridspec = gridspec or [25, 25]

    # Process arguments
    params = kwargs.get('params', None)
    sys = _create_system(sys, params)
    pointdata = [-1, 1, -1, 1] if pointdata is None else pointdata
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Create axis if needed
    user_ax = ax
    fig, ax = _process_ax_keyword(user_ax, squeeze=True, rcParams=rcParams)

    # Create copy of kwargs for later checking to find unused arguments
    initial_kwargs = dict(kwargs)

    # Utility function to create keyword arguments
    def _create_kwargs(global_kwargs, local_kwargs, **other_kwargs):
        new_kwargs = dict(global_kwargs)
        new_kwargs.update(other_kwargs)
        if isinstance(local_kwargs, dict):
            new_kwargs.update(local_kwargs)
        return new_kwargs

    # Create list for storing outputs
    out = np.array([[], None, None, None], dtype=object)

    # the maximum zorder of stramlines, vectorfield or streamplot
    flow_zorder = None

    # Plot out the main elements
    if plot_streamlines:
        kwargs_local = _create_kwargs(
            kwargs, plot_streamlines, gridspec=gridspec, gridtype=gridtype,
            ax=ax)
        out[0] += streamlines(
            sys, pointdata, timedata, _check_kwargs=False,
            suppress_warnings=suppress_warnings, **kwargs_local)

        new_zorder = max(elem.get_zorder() for elem in out[0])
        flow_zorder = max(flow_zorder, new_zorder) if flow_zorder \
            else new_zorder

        # Get rid of keyword arguments handled by streamlines
        for kw in ['arrows', 'arrow_size', 'arrow_style', 'color',
                   'dir', 'params']:
            initial_kwargs.pop(kw, None)

    # Reset the gridspec for the remaining commands, if needed
    if gridtype not in [None, 'boxgrid', 'meshgrid']:
        gridspec = None

    if plot_vectorfield:
        kwargs_local = _create_kwargs(
            kwargs, plot_vectorfield, gridspec=gridspec, ax=ax)
        out[1] = vectorfield(
            sys, pointdata, _check_kwargs=False, **kwargs_local)

        new_zorder = out[1].get_zorder()
        flow_zorder = max(flow_zorder, new_zorder) if flow_zorder \
            else new_zorder

        # Get rid of keyword arguments handled by vectorfield
        for kw in ['color', 'params']:
            initial_kwargs.pop(kw, None)

    if plot_streamplot:
        if gridtype not in [None, 'meshgrid']:
            raise ValueError(
                "gridtype must be 'meshgrid' when using streamplot")

        kwargs_local = _create_kwargs(
            kwargs, plot_streamplot, gridspec=gridspec, ax=ax)
        out[3] = streamplot(
            sys, pointdata, _check_kwargs=False, **kwargs_local)

        new_zorder = max(out[3].lines.get_zorder(), out[3].arrows.get_zorder())
        flow_zorder = max(flow_zorder, new_zorder) if flow_zorder \
            else new_zorder

        # Get rid of keyword arguments handled by streamplot
        for kw in ['color', 'params']:
            initial_kwargs.pop(kw, None)

    sep_zorder = flow_zorder + 1 if flow_zorder else None

    if plot_separatrices:
        kwargs_local = _create_kwargs(
            kwargs, plot_separatrices, gridspec=gridspec, ax=ax)
        kwargs_local['zorder'] = kwargs_local.get('zorder', sep_zorder)
        out[0] += separatrices(
            sys, pointdata, _check_kwargs=False,  **kwargs_local)

        sep_zorder = max(elem.get_zorder() for elem in out[0]) if out[0] \
            else None

        # Get rid of keyword arguments handled by separatrices
        for kw in ['arrows', 'arrow_size', 'arrow_style', 'params']:
            initial_kwargs.pop(kw, None)

    equil_zorder = sep_zorder + 1 if sep_zorder else None

    if plot_equilpoints:
        kwargs_local = _create_kwargs(
            kwargs, plot_equilpoints, gridspec=gridspec, ax=ax)
        kwargs_local['zorder'] = kwargs_local.get('zorder', equil_zorder)
        out[2] = equilpoints(
            sys, pointdata, _check_kwargs=False, **kwargs_local)

        # Get rid of keyword arguments handled by equilpoints
        for kw in ['params']:
            initial_kwargs.pop(kw, None)

    # Make sure all keyword arguments were used
    if initial_kwargs:
        raise TypeError("unrecognized keywords: ", str(initial_kwargs))

    if user_ax is None:
        if title is None:
            title = f"Phase portrait for {sys.name}"
        _update_plot_title(title, use_existing=False, rcParams=rcParams)
        ax.set_xlabel(sys.state_labels[0])
        ax.set_ylabel(sys.state_labels[1])
        plt.tight_layout()

    return ControlPlot(out, ax, fig)


def vectorfield(
        sys, pointdata, gridspec=None, zorder=None, ax=None,
        suppress_warnings=False, _check_kwargs=True, **kwargs):
    """Plot a vector field in the phase plane.

    This function plots a vector field for a two-dimensional state
    space system.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data.  If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot or an array of shape (N, 2)
        giving points of at which to plot the vector field.
    gridtype : str, optional
        The type of grid to use for generating initial conditions:
        'meshgrid' (default) generates a mesh of initial conditions within
        the specified boundaries, 'boxgrid' generates initial conditions
        along the edges of the boundary, 'circlegrid' generates a circle of
        initial conditions around each point in point data.
    gridspec : list, optional
        If the gridtype is 'meshgrid' and 'boxgrid', `gridspec` gives the
        size of the grid in the x and y axes on which to generate points.
        If gridtype is 'circlegrid', then `gridspec` is a 2-tuple
        specifying the radius and number of points around each point in the
        `pointdata` array.
    params : dict or list, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : matplotlib color spec, optional
        Plot the vector field in the given color.
    ax : `matplotlib.axes.Axes`, optional
        Use the given axes for the plot, otherwise use the current axes.

    Returns
    -------
    out : Quiver

    Other Parameters
    ----------------
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    suppress_warnings : bool, optional
        If set to True, suppress warning messages in generating trajectories.
    zorder : float, optional
        Set the zorder for the vectorfield.  In not specified, it will be
        automatically chosen by `matplotlib.axes.Axes.quiver`.

    """
    # Process keywords
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Get system parameters
    params = kwargs.pop('params', None)

    # Create system from callable, if needed
    sys = _create_system(sys, params)

    # Determine the points on which to generate the vector field
    points, _ = _make_points(pointdata, gridspec, 'meshgrid')

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Set the plotting limits
    xlim, ylim, maxlim = _set_axis_limits(ax, pointdata)

    # Figure out the color to use
    color = _get_color(kwargs, ax=ax)

    # Make sure all keyword arguments were processed
    if _check_kwargs and kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Generate phase plane (quiver) data
    vfdata = np.zeros((points.shape[0], 4))
    sys._update_params(params)
    for i, x in enumerate(points):
        vfdata[i, :2] = x
        vfdata[i, 2:] = sys._rhs(0, x, np.zeros(sys.ninputs))

    with plt.rc_context(rcParams):
        out = ax.quiver(
            vfdata[:, 0], vfdata[:, 1], vfdata[:, 2], vfdata[:, 3],
            angles='xy', color=color, zorder=zorder)

    return out


def streamplot(
        sys, pointdata, gridspec=None, zorder=None, ax=None, vary_color=False,
        vary_linewidth=False, cmap=None, norm=None, suppress_warnings=False,
        _check_kwargs=True, **kwargs):
    """Plot streamlines in the phase plane.

    This function plots the streamlines for a two-dimensional state
    space system using the `matplotlib.axes.Axes.streamplot` function.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data.  If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot.
    gridspec : list, optional
        Specifies the size of the grid in the x and y axes on which to
        generate points.
    params : dict or list, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : matplotlib color spec, optional
        Plot the vector field in the given color.
    ax : `matplotlib.axes.Axes`, optional
        Use the given axes for the plot, otherwise use the current axes.

    Returns
    -------
    out : StreamplotSet
        Containter object with lines and arrows contained in the
        streamplot. See `matplotlib.axes.Axes.streamplot` for details.

    Other Parameters
    ----------------
    cmap : str or Colormap, optional
        Colormap to use for varying the color of the streamlines.
    norm : `matplotlib.colors.Normalize`, optional
        Normalization map to use for scaling the colormap and linewidths.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.default['ctrlplot.rcParams']`.
    suppress_warnings : bool, optional
        If set to True, suppress warning messages in generating trajectories.
    vary_color : bool, optional
        If set to True, vary the color of the streamlines based on the
        magnitude of the vector field.
    vary_linewidth : bool, optional.
        If set to True, vary the linewidth of the streamlines based on the
        magnitude of the vector field.
    zorder : float, optional
        Set the zorder for the streamlines.  In not specified, it will be
        automatically chosen by `matplotlib.axes.Axes.streamplot`.

    """
    # Process keywords
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Get system parameters
    params = kwargs.pop('params', None)

    # Create system from callable, if needed
    sys = _create_system(sys, params)

    # Determine the points on which to generate the streamplot field
    points, gridspec = _make_points(pointdata, gridspec, 'meshgrid')
    grid_arr_shape = gridspec[::-1]
    xs = points[:, 0].reshape(grid_arr_shape)
    ys = points[:, 1].reshape(grid_arr_shape)

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Set the plotting limits
    xlim, ylim, maxlim = _set_axis_limits(ax, pointdata)

    # Figure out the color to use
    color = _get_color(kwargs, ax=ax)

    # Make sure all keyword arguments were processed
    if _check_kwargs and kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Generate phase plane (quiver) data
    sys._update_params(params)
    us_flat, vs_flat = np.transpose(
        [sys._rhs(0, x, np.zeros(sys.ninputs)) for x in points])
    us, vs = us_flat.reshape(grid_arr_shape), vs_flat.reshape(grid_arr_shape)

    magnitudes = np.linalg.norm([us, vs], axis=0)
    norm = norm or mpl.colors.Normalize()
    normalized = norm(magnitudes)
    cmap = plt.get_cmap(cmap)

    with plt.rc_context(rcParams):
        default_lw = plt.rcParams['lines.linewidth']
        min_lw, max_lw = 0.25*default_lw, 2*default_lw
        linewidths = normalized * (max_lw - min_lw) + min_lw \
            if vary_linewidth else None
        color = magnitudes if vary_color else color

        out = ax.streamplot(
            xs, ys, us, vs, color=color, linewidth=linewidths, cmap=cmap,
            norm=norm, zorder=zorder)

    return out


def streamlines(
        sys, pointdata, timedata=1, gridspec=None, gridtype=None, dir=None,
        zorder=None, ax=None, _check_kwargs=True, suppress_warnings=False,
        **kwargs):
    """Plot stream lines in the phase plane.

    This function plots stream lines for a two-dimensional state space
    system.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data.  If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot or an array of shape (N, 2)
        giving points of at which to plot the vector field.
    timedata : int or list of int
        Time to simulate each streamline.  If a list is given, a different
        time can be used for each initial condition in `pointdata`.
    gridtype : str, optional
        The type of grid to use for generating initial conditions:
        'meshgrid' (default) generates a mesh of initial conditions within
        the specified boundaries, 'boxgrid' generates initial conditions
        along the edges of the boundary, 'circlegrid' generates a circle of
        initial conditions around each point in point data.
    gridspec : list, optional
        If the gridtype is 'meshgrid' and 'boxgrid', `gridspec` gives the
        size of the grid in the x and y axes on which to generate points.
        If gridtype is 'circlegrid', then `gridspec` is a 2-tuple
        specifying the radius and number of points around each point in the
        `pointdata` array.
    dir : str, optional
        Direction to draw streamlines: 'forward' to flow forward in time
        from the reference points, 'reverse' to flow backward in time, or
        'both' to flow both forward and backward.  The amount of time to
        simulate in each direction is given by the `timedata` argument.
    params : dict or list, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : str
        Plot the streamlines in the given color.
    ax : `matplotlib.axes.Axes`, optional
        Use the given axes for the plot, otherwise use the current axes.

    Returns
    -------
    out : list of Line2D objects

    Other Parameters
    ----------------
    arrows : int
        Set the number of arrows to plot along the streamlines. The default
        value can be set in `config.defaults['phaseplot.arrows']`.
    arrow_size : float
        Set the size of arrows to plot along the streamlines.  The default
        value can be set in `config.defaults['phaseplot.arrow_size']`.
    arrow_style : matplotlib patch
        Set the style of arrows to plot along the streamlines.  The default
        value can be set in `config.defaults['phaseplot.arrow_style']`.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    suppress_warnings : bool, optional
        If set to True, suppress warning messages in generating trajectories.
    zorder : float, optional
        Set the zorder for the streamlines.  In not specified, it will be
        automatically chosen by `matplotlib.axes.Axes.plot`.

    """
    # Process keywords
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Get system parameters
    params = kwargs.pop('params', None)

    # Create system from callable, if needed
    sys = _create_system(sys, params)

    # Parse the arrows keyword
    arrow_pos, arrow_style = _parse_arrow_keywords(kwargs)

    # Determine the points on which to generate the streamlines
    points, gridspec = _make_points(pointdata, gridspec, gridtype=gridtype)
    if dir is None:
        dir = 'both' if gridtype == 'meshgrid' else 'forward'

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Set the axis limits
    xlim, ylim, maxlim = _set_axis_limits(ax, pointdata)

    # Figure out the color to use
    color = _get_color(kwargs, ax=ax)

    # Make sure all keyword arguments were processed
    if _check_kwargs and kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Create reverse time system, if needed
    if dir != 'forward':
        revsys = NonlinearIOSystem(
            lambda t, x, u, params: -np.asarray(sys.updfcn(t, x, u, params)),
            sys.outfcn, states=sys.nstates, inputs=sys.ninputs,
            outputs=sys.noutputs, params=sys.params)
    else:
        revsys = None

    # Generate phase plane (streamline) data
    out = []
    for i, X0 in enumerate(points):
        # Create the trajectory for this point
        timepts = _make_timepts(timedata, i)
        traj = _create_trajectory(
            sys, revsys, timepts, X0, params, dir,
            gridtype=gridtype, gridspec=gridspec, xlim=xlim, ylim=ylim,
            suppress_warnings=suppress_warnings)

        # Plot the trajectory (if there is one)
        if traj.shape[1] > 1:
            with plt.rc_context(rcParams):
                out += ax.plot(traj[0], traj[1], color=color, zorder=zorder)

                # Add arrows to the lines at specified intervals
                _add_arrows_to_line2D(
                    ax, out[-1], arrow_pos, arrowstyle=arrow_style, dir=1)
    return out


def equilpoints(
        sys, pointdata, gridspec=None, color='k', zorder=None, ax=None,
        _check_kwargs=True, **kwargs):
    """Plot equilibrium points in the phase plane.

    This function plots the equilibrium points for a planar dynamical system.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data. If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot or an array of shape (N, 2)
        giving points of at which to plot the vector field.
    gridtype : str, optional
        The type of grid to use for generating initial conditions:
        'meshgrid' (default) generates a mesh of initial conditions within
        the specified boundaries, 'boxgrid' generates initial conditions
        along the edges of the boundary, 'circlegrid' generates a circle of
        initial conditions around each point in point data.
    gridspec : list, optional
        If the gridtype is 'meshgrid' and 'boxgrid', `gridspec` gives the
        size of the grid in the x and y axes on which to generate points.
        If gridtype is 'circlegrid', then `gridspec` is a 2-tuple
        specifying the radius and number of points around each point in the
        `pointdata` array.
    params : dict or list, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : str
        Plot the equilibrium points in the given color.
    ax : `matplotlib.axes.Axes`, optional
        Use the given axes for the plot, otherwise use the current axes.

    Returns
    -------
    out : list of Line2D objects

    Other Parameters
    ----------------
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    zorder : float, optional
        Set the zorder for the equilibrium points.  In not specified, it will
        be automatically chosen by `matplotlib.axes.Axes.plot`.

    """
    # Process keywords
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Get system parameters
    params = kwargs.pop('params', None)

    # Create system from callable, if needed
    sys = _create_system(sys, params)

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Set the axis limits
    xlim, ylim, maxlim = _set_axis_limits(ax, pointdata)

    # Determine the points on which to generate the vector field
    gridspec = [5, 5] if gridspec is None else gridspec
    points, _ = _make_points(pointdata, gridspec, 'meshgrid')

    # Make sure all keyword arguments were processed
    if _check_kwargs and kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Search for equilibrium points
    equilpts = _find_equilpts(sys, points, params=params)

    # Plot the equilibrium points
    out = []
    for xeq in equilpts:
        with plt.rc_context(rcParams):
            out += ax.plot(
                xeq[0], xeq[1], marker='o', color=color, zorder=zorder)
    return out


def separatrices(
        sys, pointdata, timedata=None, gridspec=None, zorder=None, ax=None,
        _check_kwargs=True, suppress_warnings=False, **kwargs):
    """Plot separatrices in the phase plane.

    This function plots separatrices for a two-dimensional state space
    system.

    Parameters
    ----------
    sys : `NonlinearIOSystem` or callable(t, x, ...)
        I/O system or function used to generate phase plane data. If a
        function is given, the remaining arguments are drawn from the
        `params` keyword.
    pointdata : list or 2D array
        List of the form [xmin, xmax, ymin, ymax] describing the
        boundaries of the phase plot or an array of shape (N, 2)
        giving points of at which to plot the vector field.
    timedata : int or list of int
        Time to simulate each streamline.  If a list is given, a different
        time can be used for each initial condition in `pointdata`.
    gridtype : str, optional
        The type of grid to use for generating initial conditions:
        'meshgrid' (default) generates a mesh of initial conditions within
        the specified boundaries, 'boxgrid' generates initial conditions
        along the edges of the boundary, 'circlegrid' generates a circle of
        initial conditions around each point in point data.
    gridspec : list, optional
        If the gridtype is 'meshgrid' and 'boxgrid', `gridspec` gives the
        size of the grid in the x and y axes on which to generate points.
        If gridtype is 'circlegrid', then `gridspec` is a 2-tuple
        specifying the radius and number of points around each point in the
        `pointdata` array.
    params : dict or list, optional
        Parameters to pass to system. For an I/O system, `params` should be
        a dict of parameters and values. For a callable, `params` should be
        dict with key 'args' and value given by a tuple (passed to callable).
    color : matplotlib color spec, optional
        Plot the separatrices in the given color.  If a single color
        specification is given, this is used for both stable and unstable
        separatrices.  If a tuple is given, the first element is used as
        the color specification for stable separatrices and the second
        element for unstable separatrices.
    ax : `matplotlib.axes.Axes`, optional
        Use the given axes for the plot, otherwise use the current axes.

    Returns
    -------
    out : list of Line2D objects

    Other Parameters
    ----------------
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    suppress_warnings : bool, optional
        If set to True, suppress warning messages in generating trajectories.
    zorder : float, optional
        Set the zorder for the separatrices.  In not specified, it will be
        automatically chosen by `matplotlib.axes.Axes.plot`.

    Notes
    -----
    The value of `config.defaults['separatrices_radius']` is used to set the
    offset from the equilibrium point to the starting point of the separatix
    traces, in the direction of the eigenvectors evaluated at that
    equilibrium point.

    """
    # Process keywords
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # Get system parameters
    params = kwargs.pop('params', None)

    # Create system from callable, if needed
    sys = _create_system(sys, params)

    # Parse the arrows keyword
    arrow_pos, arrow_style = _parse_arrow_keywords(kwargs)

    # Determine the initial states to use in searching for equilibrium points
    gridspec = [5, 5] if gridspec is None else gridspec
    points, _ = _make_points(pointdata, gridspec, 'meshgrid')

    # Find the equilibrium points
    equilpts = _find_equilpts(sys, points, params=params)
    radius = config._get_param('phaseplot', 'separatrices_radius')

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Set the axis limits
    xlim, ylim, maxlim = _set_axis_limits(ax, pointdata)

    # Figure out the color to use for stable, unstable subspaces
    color = _get_color(kwargs)
    match color:
        case None:
            stable_color = 'r'
            unstable_color = 'b'
        case (stable_color, unstable_color) | [stable_color, unstable_color]:
            pass
        case single_color:
            stable_color = unstable_color = single_color

    # Make sure all keyword arguments were processed
    if _check_kwargs and kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    # Create a "reverse time" system to use for simulation
    revsys = NonlinearIOSystem(
        lambda t, x, u, params: -np.array(sys.updfcn(t, x, u, params)),
        sys.outfcn, states=sys.nstates, inputs=sys.ninputs,
        outputs=sys.noutputs, params=sys.params)

    # Plot separatrices by flowing backwards in time along eigenspaces
    out = []
    for i, xeq in enumerate(equilpts):
        # Figure out the linearization and eigenvectors
        evals, evecs = np.linalg.eig(sys.linearize(xeq, 0, params=params).A)

        # See if we have real eigenvalues (=> evecs are meaningful)
        if evals[0].imag > 0:
            continue

        # Create default list of time points
        if timedata is not None:
            timepts = _make_timepts(timedata, i)

        # Generate the traces
        for j, dir in enumerate(evecs.T):
            # Figure out time vector if not yet computed
            if timedata is None:
                timescale = math.log(maxlim / radius) / abs(evals[j].real)
                timepts = np.linspace(0, timescale)

            # Run the trajectory starting in eigenvector directions
            for eps in [-radius, radius]:
                x0 = xeq + dir * eps
                if evals[j].real < 0:
                    traj = _create_trajectory(
                        sys, revsys, timepts, x0, params, 'reverse',
                        gridtype='boxgrid', xlim=xlim, ylim=ylim,
                        suppress_warnings=suppress_warnings)
                    color = stable_color
                    linestyle = '--'
                elif evals[j].real > 0:
                    traj = _create_trajectory(
                        sys, revsys, timepts, x0, params, 'forward',
                        gridtype='boxgrid', xlim=xlim, ylim=ylim,
                        suppress_warnings=suppress_warnings)
                    color = unstable_color
                    linestyle = '-'

                # Plot the trajectory (if there is one)
                if traj.shape[1] > 1:
                    with plt.rc_context(rcParams):
                        out += ax.plot(
                            traj[0], traj[1], color=color,
                            linestyle=linestyle, zorder=zorder)

                    # Add arrows to the lines at specified intervals
                    with plt.rc_context(rcParams):
                        _add_arrows_to_line2D(
                            ax, out[-1], arrow_pos, arrowstyle=arrow_style,
                            dir=1)
    return out


#
# User accessible utility functions
#

# Utility function to generate boxgrid (in the form needed here)
def boxgrid(xvals, yvals):
    """Generate list of points along the edge of box.

    points = boxgrid(xvals, yvals) generates a list of points that
    corresponds to a grid given by the cross product of the x and y values.

    Parameters
    ----------
    xvals, yvals : 1D array_like
        Array of points defining the points on the lower and left edges of
        the box.

    Returns
    -------
    grid : 2D array
        Array with shape (p, 2) defining the points along the edges of the
        box, where p is the number of points around the edge.

    """
    return np.array(
        [(x, yvals[0]) for x in xvals[:-1]] +           # lower edge
        [(xvals[-1], y) for y in yvals[:-1]] +          # right edge
        [(x, yvals[-1]) for x in xvals[:0:-1]] +        # upper edge
        [(xvals[0], y) for y in yvals[:0:-1]]           # left edge
    )


# Utility function to generate meshgrid (in the form needed here)
# TODO: add examples of using grid functions directly
def meshgrid(xvals, yvals):
    """Generate list of points forming a mesh.

    points = meshgrid(xvals, yvals) generates a list of points that
    corresponds to a grid given by the cross product of the x and y values.

    Parameters
    ----------
    xvals, yvals : 1D array_like
        Array of points defining the points on the lower and left edges of
        the box.

    Returns
    -------
    grid : 2D array
        Array of points with shape (n * m, 2) defining the mesh.

    """
    xvals, yvals = np.meshgrid(xvals, yvals)
    grid = np.zeros((xvals.shape[0] * xvals.shape[1], 2))
    grid[:, 0] = xvals.reshape(-1)
    grid[:, 1] = yvals.reshape(-1)

    return grid


# Utility function to generate circular grid
def circlegrid(centers, radius, num):
    """Generate list of points around a circle.

    points = circlegrid(centers, radius, num) generates a list of points
    that form a circle around a list of centers.

    Parameters
    ----------
    centers : 2D array_like
        Array of points with shape (p, 2) defining centers of the circles.
    radius : float
        Radius of the points to be generated around each center.
    num : int
        Number of points to generate around the circle.

    Returns
    -------
    grid : 2D array
        Array of points with shape (p * num, 2) defining the circles.

    """
    centers = np.atleast_2d(np.array(centers))
    grid = np.zeros((centers.shape[0] * num, 2))
    for i, center in enumerate(centers):
        grid[i * num: (i + 1) * num, :] = center + np.array([
            [radius * math.cos(theta), radius * math.sin(theta)] for
            theta in np.linspace(0, 2 * math.pi, num, endpoint=False)])
    return grid


#
# Internal utility functions
#

# Create a system from a callable
def _create_system(sys, params):
    if isinstance(sys, NonlinearIOSystem):
        if sys.nstates != 2:
            raise ValueError("system must be planar")
        return sys

    # Make sure that if params is present, it has 'args' key
    if params and not params.get('args', None):
        raise ValueError("params must be dict with key 'args'")

    _update = lambda t, x, u, params: sys(t, x, *params.get('args', ()))
    _output = lambda t, x, u, params: np.array([])
    return NonlinearIOSystem(
        _update, _output, states=2, inputs=0, outputs=0, name="_callable")


# Set axis limits for the plot
def _set_axis_limits(ax, pointdata):
    # Get the current axis limits
    if ax.lines:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        # Nothing on the plot => always use new limits
        xlim, ylim = [np.inf, -np.inf], [np.inf, -np.inf]

    # Short utility function for updating axis limits
    def _update_limits(cur, new):
        return [min(cur[0], np.min(new)), max(cur[1], np.max(new))]

    # If we were passed a box, use that to update the limits
    if isinstance(pointdata, list) and len(pointdata) == 4:
        xlim = _update_limits(xlim, [pointdata[0], pointdata[1]])
        ylim = _update_limits(ylim, [pointdata[2], pointdata[3]])

    elif isinstance(pointdata, np.ndarray):
        pointdata = np.atleast_2d(pointdata)
        xlim = _update_limits(
            xlim, [np.min(pointdata[:, 0]), np.max(pointdata[:, 0])])
        ylim = _update_limits(
            ylim, [np.min(pointdata[:, 1]), np.max(pointdata[:, 1])])

    # Keep track of the largest dimension on the plot
    maxlim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    # Set the new limits
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return xlim, ylim, maxlim


# Find equilibrium points
def _find_equilpts(sys, points, params=None):
    equilpts = []
    for i, x0 in enumerate(points):
        # Look for an equilibrium point near this point
        xeq, ueq = find_operating_point(sys, x0, 0, params=params)

        if xeq is None:
            continue            # didn't find anything

        # See if we have already found this point
        seen = False
        for x in equilpts:
            if np.allclose(np.array(x), xeq):
                seen = True
        if seen:
            continue

        # Save a new point
        equilpts += [xeq.tolist()]

    return equilpts


def _make_points(pointdata, gridspec, gridtype):
    # Check to see what type of data we got
    if isinstance(pointdata, np.ndarray) and gridtype is None:
        pointdata = np.atleast_2d(pointdata)
        if pointdata.shape[1] == 2:
            # Given a list of points => no action required
            return pointdata, None

    # Utility function to parse (and check) input arguments
    def _parse_args(defsize):
        if gridspec is None:
            return defsize

        elif not isinstance(gridspec, (list, tuple)) or \
             len(gridspec) != len(defsize):
            raise ValueError("invalid grid specification")

        return gridspec

    # Generate points based on grid type
    match gridtype:
        case 'boxgrid' | None:
            gridspec = _parse_args([6, 4])
            points = boxgrid(
                np.linspace(pointdata[0], pointdata[1], gridspec[0]),
                np.linspace(pointdata[2], pointdata[3], gridspec[1]))

        case 'meshgrid':
            gridspec = _parse_args([9, 6])
            points = meshgrid(
                np.linspace(pointdata[0], pointdata[1], gridspec[0]),
                np.linspace(pointdata[2], pointdata[3], gridspec[1]))

        case 'circlegrid':
            gridspec = _parse_args((0.5, 10))
            if isinstance(pointdata, np.ndarray):
                # Create circles around each point
                points = circlegrid(pointdata, gridspec[0], gridspec[1])
            else:
                # Create circle around center of the plot
                points = circlegrid(
                    np.array(
                        [(pointdata[0] + pointdata[1]) / 2,
                         (pointdata[0] + pointdata[1]) / 2]),
                    gridspec[0], gridspec[1])

        case _:
            raise ValueError(f"unknown grid type '{gridtype}'")

    return points, gridspec


def _parse_arrow_keywords(kwargs):
    # Get values for params (and pop from list to allow keyword use in plot)
    # TODO: turn this into a utility function (shared with nyquist_plot?)
    arrows = config._get_param(
        'phaseplot', 'arrows', kwargs, None, pop=True)
    arrow_size = config._get_param(
        'phaseplot', 'arrow_size', kwargs, None, pop=True)
    arrow_style = config._get_param('phaseplot', 'arrow_style', kwargs, None)

    # Parse the arrows keyword
    if not arrows:
        arrow_pos = []
    elif isinstance(arrows, int):
        N = arrows
        # Space arrows out, starting midway along each "region"
        arrow_pos = np.linspace(0.5/N, 1 + 0.5/N, N, endpoint=False)
    elif isinstance(arrows, (list, np.ndarray)):
        arrow_pos = np.sort(np.atleast_1d(arrows))
    else:
        raise ValueError("unknown or unsupported arrow location")

    # Set the arrow style
    if arrow_style is None:
        arrow_style = mpl.patches.ArrowStyle(
            'simple', head_width=int(2 * arrow_size / 3),
            head_length=arrow_size)

    return arrow_pos, arrow_style


# TODO: move to ctrlplot?
def _create_trajectory(
        sys, revsys, timepts, X0, params, dir, suppress_warnings=False,
        gridtype=None, gridspec=None, xlim=None, ylim=None):
    # Compute the forward trajectory
    if dir == 'forward' or dir == 'both':
        fwdresp = input_output_response(
            sys, timepts, initial_state=X0, params=params, ignore_errors=True)
        if not fwdresp.success and not suppress_warnings:
            warnings.warn(f"initial_state={X0}, {fwdresp.message}")

    # Compute the reverse trajectory
    if dir == 'reverse' or dir == 'both':
        revresp = input_output_response(
            revsys, timepts, initial_state=X0, params=params,
            ignore_errors=True)
        if not revresp.success and not suppress_warnings:
            warnings.warn(f"initial_state={X0}, {revresp.message}")

    # Create the trace to plot
    if dir == 'forward':
        traj = fwdresp.states
    elif dir == 'reverse':
        traj = revresp.states[:, ::-1]
    elif dir == 'both':
        traj = np.hstack([revresp.states[:, :1:-1], fwdresp.states])

    # Remove points outside the window (keep first point beyond boundary)
    inrange = np.asarray(
        (traj[0] >= xlim[0]) & (traj[0] <= xlim[1]) &
        (traj[1] >= ylim[0]) & (traj[1] <= ylim[1]))
    inrange[:-1] = inrange[:-1] | inrange[1:]   # keep if next point in range
    inrange[1:] = inrange[1:] | inrange[:-1]    # keep if prev point in range

    return traj[:, inrange]


def _make_timepts(timepts, i):
    if timepts is None:
        return np.linspace(0, 1)
    elif isinstance(timepts, (int, float)):
        return np.linspace(0, timepts)
    elif timepts.ndim == 2:
        return timepts[i]
    return timepts


#
# Legacy phase plot function
#
# Author: Richard Murray
# Date: 24 July 2011, converted from MATLAB version (2002); based on
# a version by Kristi Morgansen
#
def phase_plot(odefun, X=None, Y=None, scale=1, X0=None, T=None,
               lingrid=None, lintime=None, logtime=None, timepts=None,
               parms=None, params=(), tfirst=False, verbose=True):

    """(legacy) Phase plot for 2D dynamical systems.

    .. deprecated:: 0.10.1
        This function is deprecated; use `phase_plane_plot` instead.

    Produces a vector field or stream line plot for a planar system.  This
    function has been replaced by the `phase_plane_map` and
    `phase_plane_plot` functions.

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
        Computes the time derivative of y (compatible with odeint).  The
        function should be the same for as used for `scipy.integrate`.
        Namely, it should be a function of the form dx/dt = F(t, x) that
        accepts a state x of dimension 2 and returns a derivative dx/dt of
        dimension 2.
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
    T: array_like or number, optional
        Length of time to run simulations that generate streamlines.
        If a single number, the same simulation time is used for all
        initial conditions.  Otherwise, should be a list of length
        len(X0) that gives the simulation time for each initial
        condition.  Default value = 50.
    lingrid : integer or 2-tuple of integers, optional
        Argument is either N or (N, M).  If X0 is given and X, Y are
        missing, a grid of arrows is produced using the limits of the
        initial conditions, with N grid points in each dimension or N grid
        points in x and M grid points in y.
    lintime : integer or tuple (integer, float), optional
        If a single integer N is given, draw N arrows using equally space
        time points.  If a tuple (N, lambda) is given, draw N arrows using
        exponential time constant lambda
    timepts : array_like, optional
        Draw arrows at the given list times [t1, t2, ...]
    tfirst : bool, optional
        If True, call `func` with signature ``func(t, x, ...)``.
    params: tuple, optional
        List of parameters to pass to vector field: ``func(x, t, *params)``.

    See Also
    --------
    box_grid

    """
    # Generate a deprecation warning
    warnings.warn(
        "phase_plot() is deprecated; use phase_plane_plot() instead",
        FutureWarning)

    #
    # Figure out ranges for phase plot (argument processing)
    #
    #! TODO: need to add error checking to arguments
    #! TODO: think through proper action if multiple options are given
    #
    autoFlag = False
    logtimeFlag = False
    timeptsFlag = False
    Narrows = 0

    # Get parameters to pass to function
    if parms:
        warnings.warn(
            "keyword 'parms' is deprecated; use 'params'", FutureWarning)
        if params:
            raise ControlArgument("duplicate keywords 'parms' and 'params'")
        else:
            params = parms

    if lingrid is not None:
        autoFlag = True
        Narrows = lingrid
        if (verbose):
            print('Using auto arrows\n')

    elif logtime is not None:
        logtimeFlag = True
        Narrows = logtime[0]
        timefactor = logtime[1]
        if (verbose):
            print('Using logtime arrows\n')

    elif timepts is not None:
        timeptsFlag = True
        Narrows = len(timepts)

    # Figure out the set of points for the quiver plot
    #! TODO: Add sanity checks
    elif X is not None and Y is not None:
        x1, x2 = np.meshgrid(
            np.linspace(X[0], X[1], X[2]),
            np.linspace(Y[0], Y[1], Y[2]))
        Narrows = len(x1)

    else:
        # If we weren't given any grid points, don't plot arrows
        Narrows = 0

    if not autoFlag and not logtimeFlag and not timeptsFlag and Narrows > 0:
        # Now calculate the vector field at those points
        (nr,nc) = x1.shape
        dx = np.empty((nr, nc, 2))
        for i in range(nr):
            for j in range(nc):
                if tfirst:
                    dx[i, j, :] = np.squeeze(
                        odefun(0, [x1[i,j], x2[i,j]], *params))
                else:
                    dx[i, j, :] = np.squeeze(
                        odefun([x1[i,j], x2[i,j]], 0, *params))

        # Plot the quiver plot
        #! TODO: figure out arguments to make arrows show up correctly
        if scale is None:
            plt.quiver(x1, x2, dx[:,:,1], dx[:,:,2], angles='xy')
        elif (scale != 0):
            plt.quiver(x1, x2, dx[:,:,0]*np.abs(scale),
                       dx[:,:,1]*np.abs(scale), angles='xy')
            #! TODO: optimize parameters for arrows
            #! TODO: figure out arguments to make arrows show up correctly
            # xy = plt.quiver(...)
            # set(xy, 'LineWidth', PP_arrow_linewidth, 'Color', 'b')

        #! TODO: Tweak the shape of the plot
        # a=gca; set(a,'DataAspectRatio',[1,1,1])
        # set(a,'XLim',X(1:2)); set(a,'YLim',Y(1:2))
        plt.xlabel('x1'); plt.ylabel('x2')

    # See if we should also generate the streamlines
    if X0 is None or len(X0) == 0:
        return

    # Convert initial conditions to a numpy array
    X0 = np.array(X0)
    (nr, nc) = np.shape(X0)

    # Generate some empty matrices to keep arrow information
    x1 = np.empty((nr, Narrows))
    x2 = np.empty((nr, Narrows))
    dx = np.empty((nr, Narrows, 2))

    # See if we were passed a simulation time
    if T is None:
        T = 50

    # Parse the time we were passed
    TSPAN = T
    if isinstance(T, (int, float)):
        TSPAN = np.linspace(0, T, 100)

    # Figure out the limits for the plot
    if scale is None:
        # Assume that the current axis are set as we want them
        alim = plt.axis()
        xmin = alim[0]; xmax = alim[1]
        ymin = alim[2]; ymax = alim[3]
    else:
        # Use the maximum extent of all trajectories
        xmin = np.min(X0[:,0]); xmax = np.max(X0[:,0])
        ymin = np.min(X0[:,1]); ymax = np.max(X0[:,1])

    # Generate the streamlines for each initial condition
    for i in range(nr):
        state = odeint(odefun, X0[i], TSPAN, args=params, tfirst=tfirst)
        time = TSPAN

        plt.plot(state[:,0], state[:,1])
        #! TODO: add back in colors for stream lines
        # PP_stream_color(np.mod(i-1, len(PP_stream_color))+1))
        # set(h[i], 'LineWidth', PP_stream_linewidth)

        # Plot arrows if quiver parameters were 'auto'
        if autoFlag or logtimeFlag or timeptsFlag:
            # Compute the locations of the arrows
            #! TODO: check this logic to make sure it works in python
            for j in range(Narrows):

                # Figure out starting index; headless arrows start at 0
                k = -1 if scale is None else 0

                # Figure out what time index to use for the next point
                if autoFlag:
                    # Use a linear scaling based on ODE time vector
                    tind = np.floor((len(time)/Narrows) * (j-k)) + k
                elif logtimeFlag:
                    # Use an exponential time vector
                    # MATLAB: tind = find(time < (j-k) / lambda, 1, 'last')
                    tarr = _find(time < (j-k) / timefactor)
                    tind = tarr[-1] if len(tarr) else 0
                elif timeptsFlag:
                    # Use specified time points
                    # MATLAB: tind = find(time < Y[j], 1, 'last')
                    tarr = _find(time < timepts[j])
                    tind = tarr[-1] if len(tarr) else 0

                # For tailless arrows, skip the first point
                if tind == 0 and scale is None:
                    continue

                # Figure out the arrow at this point on the curve
                x1[i,j] = state[tind, 0]
                x2[i,j] = state[tind, 1]

                # Skip arrows outside of initial condition box
                if (scale is not None or
                     (x1[i,j] <= xmax and x1[i,j] >= xmin and
                      x2[i,j] <= ymax and x2[i,j] >= ymin)):
                    if tfirst:
                        pass
                        v = odefun(0, [x1[i,j], x2[i,j]], *params)
                    else:
                        v = odefun([x1[i,j], x2[i,j]], 0, *params)
                    dx[i, j, 0] = v[0]; dx[i, j, 1] = v[1]
                else:
                    dx[i, j, 0] = 0; dx[i, j, 1] = 0

    # Set the plot shape before plotting arrows to avoid warping
    # a=gca
    # if (scale != None):
    #     set(a,'DataAspectRatio', [1,1,1])
    # if (xmin != xmax and ymin != ymax):
    #     plt.axis([xmin, xmax, ymin, ymax])
    # set(a, 'Box', 'on')

    # Plot arrows on the streamlines
    if scale is None and Narrows > 0:
        # Use a tailless arrow
        #! TODO: figure out arguments to make arrows show up correctly
        plt.quiver(x1, x2, dx[:,:,0], dx[:,:,1], angles='xy')
    elif scale != 0 and Narrows > 0:
        plt.quiver(x1, x2, dx[:,:,0]*abs(scale), dx[:,:,1]*abs(scale),
                   angles='xy')
        #! TODO: figure out arguments to make arrows show up correctly
        # xy = plt.quiver(...)
        # set(xy, 'LineWidth', PP_arrow_linewidth)
        # set(xy, 'AutoScale', 'off')
        # set(xy, 'AutoScaleFactor', 0)

    if scale < 0:
        plt.plot(x1, x2, 'b.');        # add dots at base
        # bp = plt.plot(...)
        # set(bp, 'MarkerSize', PP_arrow_markersize)


# Utility function for generating initial conditions around a box
def box_grid(xlimp, ylimp):
    """Generate list of points on edge of box.

    .. deprecated:: 0.10.0
        Use `phaseplot.boxgrid` instead.

    list = box_grid([xmin xmax xnum], [ymin ymax ynum]) generates a
    list of points that correspond to a uniform grid at the end of the
    box defined by the corners [xmin ymin] and [xmax ymax].

    """

    # Generate a deprecation warning
    warnings.warn(
        "box_grid() is deprecated; use phaseplot.boxgrid() instead",
        FutureWarning)

    return boxgrid(
        np.linspace(xlimp[0], xlimp[1], xlimp[2]),
        np.linspace(ylimp[0], ylimp[1], ylimp[2]))


# TODO: rename to something more useful (or remove??)
def _find(condition):
    """Returns indices where ravel(a) is true.

    Private implementation of deprecated `matplotlib.mlab.find`.

    """
    return np.nonzero(np.ravel(condition))[0]
