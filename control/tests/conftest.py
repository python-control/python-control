# contest.py - pytest local plugins and fixtures

import control
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def use_numpy_ndarray():
    """Switch the config to use ndarray instead of matrix"""
    if os.getenv("PYTHON_CONTROL_STATESPACE_ARRAY") == "1":
        control.config.defaults['statesp.use_numpy_matrix'] = False
