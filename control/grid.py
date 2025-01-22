# grid.py - code to add gridlines to root locus and pole-zero diagrams

"""Functions to add gridlines to root locus and pole-zero diagrams.

This code generates grids for pole-zero diagrams (including root locus
diagrams).  Rather than just draw a grid in place, it uses the
AxisArtist package to generate a custom grid that will scale with the
figure.

"""

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.angle_helper as angle_helper
import numpy as np
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist.grid_helper_curvelinear import \
    GridHelperCurveLinear
from numpy import cos, exp, linspace, pi, sin, sqrt

from .iosys import isdtime


class FormatterDMS():
    """Transforms angle ticks to damping ratios."""
    def __call__(self, direction, factor, values):
        angles_deg = np.asarray(values)/factor
        damping_ratios = np.cos((180-angles_deg) * np.pi/180)
        ret = ["%.2f" % val for val in damping_ratios]
        return ret


class ModifiedExtremeFinderCycle(angle_helper.ExtremeFinderCycle):
    """Changed to allow only left hand-side polar grid.

    https://matplotlib.org/_modules/mpl_toolkits/axisartist/angle_helper.html#ExtremeFinderCycle.__call__
    """
    def __call__(self, transform_xy, x1, y1, x2, y2):
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        with np.errstate(invalid='ignore'):
            if self.lon_cycle is not None:
                lon0 = np.nanmin(lon)
                # Changed from 180 to 360 to be able to span only
                # 90-270 (left hand side)
                lon -= 360. * ((lon - lon0) > 360.)
            if self.lat_cycle is not None:  # pragma: no cover
                lat0 = np.nanmin(lat)
                lat -= 360. * ((lat - lat0) > 180.)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        # check cycle
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        if self.lat_cycle:  # pragma: no cover
            lat_max = min(lat_max, lat_min + self.lat_cycle)

        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)

        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)

        return lon_min, lon_max, lat_min, lat_max


def sgrid(subplot=(1, 1, 1), scaling=None):
    # From matplotlib demos:
    # https://matplotlib.org/gallery/axisartist/demo_curvelinear_grid.html
    # https://matplotlib.org/gallery/axisartist/demo_floating_axis.html

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degrees
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    sampling_points = 20
    extreme_finder = ModifiedExtremeFinderCycle(
        sampling_points, sampling_points, lon_cycle=360, lat_cycle=None,
        lon_minmax=(90, 270), lat_minmax=(0, np.inf),)

    grid_locator1 = angle_helper.LocatorDMS(15)
    tick_formatter1 = FormatterDMS()
    grid_helper = GridHelperCurveLinear(
        tr, extreme_finder=extreme_finder, grid_locator1=grid_locator1,
        tick_formatter1=tick_formatter1)

    # Set up an axes with a specialized grid helper
    fig = plt.gcf()
    ax = SubplotHost(fig, *subplot, grid_helper=grid_helper)

    # make ticklabels of right invisible, and top axis visible.
    ax.axis[:].major_ticklabels.set_visible(True)
    ax.axis[:].major_ticks.set_visible(False)
    ax.axis[:].invert_ticklabel_direction()
    ax.axis[:].major_ticklabels.set_color('gray')

    # Set up internal tickmarks and labels along the real/imag axes
    ax.axis["wnxneg"] = axis = ax.new_floating_axis(0, 180)
    axis.set_ticklabel_direction("-")
    axis.label.set_visible(False)

    ax.axis["wnxpos"] = axis = ax.new_floating_axis(0, 0)
    axis.label.set_visible(False)

    ax.axis["wnypos"] = axis = ax.new_floating_axis(0, 90)
    axis.label.set_visible(False)
    axis.set_axis_direction("right")

    ax.axis["wnyneg"] = axis = ax.new_floating_axis(0, 270)
    axis.label.set_visible(False)
    axis.set_axis_direction("left")
    axis.invert_ticklabel_direction()
    axis.set_ticklabel_direction("-")

    # let left axis shows ticklabels for 1st coordinate (angle)
    ax.axis["left"].get_helper().nth_coord_ticks = 0
    ax.axis["right"].get_helper().nth_coord_ticks = 0
    ax.axis["left"].get_helper().nth_coord_ticks = 0
    ax.axis["bottom"].get_helper().nth_coord_ticks = 0

    fig.add_subplot(ax)
    ax.grid(True, zorder=0, linestyle='dotted')

    _final_setup(ax, scaling=scaling)
    return ax, fig


# If not grid is given, at least separate stable/unstable regions
def nogrid(dt=None, ax=None, scaling=None):
    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    # Draw the unit circle for discrete-time systems
    if isdtime(dt=dt, strict=True):
        s = np.linspace(0, 2*pi, 100)
        ax.plot(np.cos(s), np.sin(s), 'k--', lw=0.5, dashes=(5, 5))

    _final_setup(ax, scaling=scaling)
    return ax, fig

# Grid for discrete-time system (drawn, not rendered by AxisArtist)
# TODO (at some point): think about using customized grid generator?
def zgrid(zetas=None, wns=None, ax=None, scaling=None):
    """Draws discrete damping and frequency grid"""

    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    # Constant damping lines
    if zetas is None:
        zetas = linspace(0, 0.9, 10)
    for zeta in zetas:
        # Calculate in polar coordinates
        factor = zeta/sqrt(1-zeta**2)
        x = linspace(0, sqrt(1-zeta**2), 200)
        ang = pi*x
        mag = exp(-pi*factor*x)
        # Draw upper part in rectangular coordinates
        xret = mag*cos(ang)
        yret = mag*sin(ang)
        ax.plot(xret, yret, ':', color='grey', lw=0.75)
        # Draw lower part in rectangular coordinates
        xret = mag*cos(-ang)
        yret = mag*sin(-ang)
        ax.plot(xret, yret, ':', color='grey', lw=0.75)
        # Annotation
        an_i = int(len(xret)/2.5)
        an_x = xret[an_i]
        an_y = yret[an_i]
        ax.annotate(str(round(zeta, 2)), xy=(an_x, an_y),
                    xytext=(an_x, an_y), size=7)

    # Constant natural frequency lines
    if wns is None:
        wns = linspace(0, 1, 10)
    for a in wns:
        # Calculate in polar coordinates
        x = linspace(-pi/2, pi/2, 200)
        ang = pi*a*sin(x)
        mag = exp(-pi*a*cos(x))
        # Draw in rectangular coordinates
        xret = mag*cos(ang)
        yret = mag*sin(ang)
        ax.plot(xret, yret, ':', color='grey', lw=0.75)
        # Annotation
        an_i = -1
        an_x = xret[an_i]
        an_y = yret[an_i]
        num = '{:1.1f}'.format(a)
        ax.annotate(r"$\frac{"+num+r"\pi}{T}$", xy=(an_x, an_y),
                    xytext=(an_x, an_y), size=9)

    # Set default axes to allow some room around the unit circle
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    _final_setup(ax, scaling=scaling)
    return ax, fig


# Utility function used by all grid code
def _final_setup(ax, scaling=None):
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(y=0, color='black', lw=0.25)
    ax.axvline(x=0, color='black', lw=0.25)

    # Set up the scaling for the axes
    scaling = 'equal' if scaling is None else scaling
    plt.axis(scaling)
