import numpy as np
from numpy import cos, sin, sqrt, linspace, pi, exp
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist.grid_helper_curvelinear \
    import GridHelperCurveLinear
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D


class FormatterDMS(object):
    '''Transforms angle ticks to damping ratios'''
    def __call__(self, direction, factor, values):
        angles_deg = np.asarray(values)/factor
        damping_ratios = np.cos((180-angles_deg) * np.pi/180)
        ret = ["%.2f" % val for val in damping_ratios]
        return ret


class ModifiedExtremeFinderCycle(angle_helper.ExtremeFinderCycle):
    '''Changed to allow only left hand-side polar grid

    https://matplotlib.org/_modules/mpl_toolkits/axisartist/angle_helper.html#ExtremeFinderCycle.__call__
    '''
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


def sgrid():
    # From matplotlib demos:
    # https://matplotlib.org/gallery/axisartist/demo_curvelinear_grid.html
    # https://matplotlib.org/gallery/axisartist/demo_floating_axis.html

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
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

    fig = plt.gcf()
    ax = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    # make ticklabels of right invisible, and top axis visible.
    visible = True
    ax.axis[:].major_ticklabels.set_visible(visible)
    ax.axis[:].major_ticks.set_visible(False)
    ax.axis[:].invert_ticklabel_direction()

    ax.axis["wnxneg"] = axis = ax.new_floating_axis(0, 180)
    axis.set_ticklabel_direction("-")
    axis.label.set_visible(False)
    ax.axis["wnxpos"] = axis = ax.new_floating_axis(0, 0)
    axis.label.set_visible(False)
    ax.axis["wnypos"] = axis = ax.new_floating_axis(0, 90)
    axis.label.set_visible(False)
    axis.set_axis_direction("left")
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

    # RECTANGULAR X Y AXES WITH SCALE
    # par2 = ax.twiny()
    # par2.axis["top"].toggle(all=False)
    # par2.axis["right"].toggle(all=False)
    # new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    # par2.axis["left"] = new_fixed_axis(loc="left",
    #                                   axes=par2,
    #                                   offset=(0, 0))
    # par2.axis["bottom"] = new_fixed_axis(loc="bottom",
    #                                     axes=par2,
    #                                     offset=(0, 0))
    # FINISH RECTANGULAR

    ax.grid(True, zorder=0, linestyle='dotted')

    _final_setup(ax)
    return ax, fig


def _final_setup(ax):
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(y=0, color='black', lw=1)
    ax.axvline(x=0, color='black', lw=1)
    plt.axis('equal')


def nogrid():
    f = plt.gcf()
    ax = plt.axes()

    _final_setup(ax)
    return ax, f


def zgrid(zetas=None, wns=None, ax=None):
    '''Draws discrete damping and frequency grid'''

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
        # Draw upper part in retangular coordinates
        xret = mag*cos(ang)
        yret = mag*sin(ang)
        ax.plot(xret, yret, ':', color='grey', lw=0.75)
        # Draw lower part in retangular coordinates
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
        # Draw in retangular coordinates
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

    _final_setup(ax)
    return ax, fig
