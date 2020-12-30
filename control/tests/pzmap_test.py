# -*- coding: utf-8 -*-
""" pzmap_test.py - test pzmap()

Created on Thu Aug 20 20:06:21 2020

@author: bnavigator
"""

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist import Axes as mpltAxes

from control import TransferFunction, config, pzmap


@pytest.mark.parametrize("kwargs",
                         [pytest.param(dict(), id="default"),
                          pytest.param(dict(plot=False), id="plot=False"),
                          pytest.param(dict(plot=True), id="plot=True"),
                          pytest.param(dict(grid=True), id="grid=True"),
                          pytest.param(dict(title="My Title"), id="title")])
@pytest.mark.parametrize("setdefaults", [False, True], ids=["kw", "config"])
@pytest.mark.parametrize("dt", [0, 1], ids=["s", "z"])
def test_pzmap(kwargs, setdefaults, dt, editsdefaults, mplcleanup):
    """Test pzmap"""
    # T from from pvtol-nested example
    T = TransferFunction([-9.0250000e-01, -4.7200750e+01, -8.6812900e+02,
                          +5.6261850e+03, +2.1258472e+05, +8.4724600e+05,
                          +1.0192000e+06, +2.3520000e+05],
                         [9.02500000e-03, 9.92862812e-01, 4.96974094e+01,
                          1.35705659e+03, 2.09294163e+04, 1.64898435e+05,
                          6.54572220e+05, 1.25274600e+06, 1.02420000e+06,
                          2.35200000e+05],
                         dt)

    Pref = [-23.8877+19.3837j, -23.8877-19.3837j, -23.8349+15.7846j,
            -23.8349-15.7846j,  -5.2320 +0.4117j,  -5.2320 -0.4117j,
            -2.2246 +0.0000j,  -1.5160 +0.0000j,  -0.3627 +0.0000j]
    Zref = [-23.8877+19.3837j, -23.8877-19.3837j, +14.3637 +0.0000j,
            -14.3637 +0.0000j,  -2.2246 +0.0000j,  -2.0000 +0.0000j,
            -0.3000 +0.0000j]

    pzkwargs = kwargs.copy()
    if setdefaults:
        for k in ['plot', 'grid']:
            if k in pzkwargs:
                v = pzkwargs.pop(k)
                config.set_defaults('pzmap', **{k: v})

    P, Z = pzmap(T, **pzkwargs)

    np.testing.assert_allclose(P, Pref, rtol=1e-3)
    np.testing.assert_allclose(Z, Zref, rtol=1e-3)

    if kwargs.get('plot', True):
        ax = plt.gca()

        assert ax.get_title() == kwargs.get('title', 'Pole Zero Map')

        # FIXME: This won't work when zgrid and sgrid are unified
        children = ax.get_children()
        has_zgrid = False
        for c in children:
            if isinstance(c, matplotlib.text.Annotation):
                if r'\pi' in c.get_text():
                    has_zgrid = True
        has_sgrid = isinstance(ax, mpltAxes)

        if kwargs.get('grid', False):
            assert dt == has_zgrid
            assert dt != has_sgrid
        else:
            assert not has_zgrid
            assert not has_sgrid
    else:
        assert not plt.get_fignums()


def test_pzmap_warns():
    with pytest.warns(FutureWarning):
        pzmap(TransferFunction([1], [1, 2]), Plot=True)


def test_pzmap_raises():
    with pytest.raises(TypeError):
        # not an LTI system
        pzmap(([1], [1,2]))
