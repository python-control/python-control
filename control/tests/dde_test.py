import numpy as np
import pytest
import matplotlib.pyplot as plt

from control.dde import *
from control.xferfcn import tf


s = tf('s')
