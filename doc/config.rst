.. currentmodule:: control
.. _package-configuration-parameters:

Package configuration parameters
================================

The python-control library can be customized to allow for different default
values for selected parameters.  This includes the ability to set the style
for various types of plots and establishing the underlying representation for
state space matrices.

To set the default value of a configuration variable, set the appropriate
element of the `config.defaults` dictionary::

    ct.config.defaults['module.parameter'] = value

The :func:`set_defaults` function can also be used to
set multiple configuration parameters at the same time::

    ct.set_defaults('module', param1=val1, param2=val2, ...]

Finally, there are also functions available set collections of variables based
on standard configurations.

Selected variables that can be configured, along with their default values:

  * freqplot.dB (False): Bode plot magnitude plotted in dB (otherwise powers
    of 10)

  * freqplot.deg (True): Bode plot phase plotted in degrees (otherwise radians)

  * freqplot.Hz (False): Bode plot frequency plotted in Hertz (otherwise
    rad/sec)

  * freqplot.grid (True): Include grids for magnitude and phase plots

  * freqplot.number_of_samples (1000): Number of frequency points in Bode plots

  * freqplot.feature_periphery_decade (1.0): How many decades to include in
    the frequency range on both sides of features (poles, zeros).

  * statesp.default_dt and xferfcn.default_dt (None): set the default value
    of dt when constructing new LTI systems

  * statesp.remove_useless_states (True): remove states that have no effect
    on the input-output dynamics of the system

Additional parameter variables are documented in individual functions

Functions that can be used to set standard configurations:

.. autosummary::

   reset_defaults
   use_fbs_defaults
   use_matlab_defaults
   use_legacy_defaults
