.. _plotting-module:

.. currentmodule:: control

**********************************
Input/output Response and Plotting
**********************************

The Python Control Systems Toolbox contains a number of functions for
computing and plotting input/output responses in the time and
frequency domain, root locus diagrams, and other standard charts used
in control system analysis, for example::

  bode_plot(sys)
  nyquist_plot([sys1, sys2])
  phase_plane_plot(sys, limits)
  pole_zero_plot(sys)
  root_locus_plot(sys)

While plotting functions can be called directly, the standard pattern used
in the toolbox is to provide a function that performs the basic computation
or analysis (e.g., computation of the time or frequency response) and
returns and object representing the output data.  A separate plotting
function, typically ending in `_plot` is then used to plot the data,
resulting in the following standard pattern::

  response = ct.nyquist_response([sys1, sys2])
  count = ct.response.count          # number of encirclements of -1
  cplt = ct.nyquist_plot(response)  # Nyquist plot

Plotting commands return a :class:`ControlPlot` object that
provides access to the individual lines in the generated plot using
`cplt.lines`, allowing various aspects of the plot to be modified to
suit specific needs.

The plotting function is also available via the `plot()` method of the
analysis object, allowing the following type of calls::

  step_response(sys).plot()
  frequency_response(sys).plot()
  nyquist_response(sys).plot()
  pp.streamlines(sys, limits).plot()
  root_locus_map(sys).plot()

The remainder of this chapter provides additional documentation on how
these response and plotting functions can be customized.


Time response data
==================

LTI response functions
----------------------

A number of functions are available for computing the output (and
state) response of an LTI systems:

.. autosummary::

   initial_response
   step_response
   impulse_response
   forced_response

Each of these functions returns a :class:`TimeResponseData` object
that contains the data for the time response (described in more detail
in the next section).

The :func:`forced_response` system is the most general and allows by
the zero initial state response to be simulated as well as the
response from a non-zero initial condition.

For linear time invariant (LTI) systems, the :func:`impulse_response`,
:func:`initial_response`, and :func:`step_response` functions will
automatically compute the time vector based on the poles and zeros of
the system.  If a list of systems is passed, a common time vector will be
computed and a list of responses will be returned in the form of a
:class:`TimeResponseList` object.  The :func:`forced_response` function can
also take a list of systems, to which a single common input is applied.
The :class:`TimeResponseList` object has a `plot()` method that will plot
each of the responses in turn, using a sequence of different colors with
appropriate titles and legends.

In addition the :func:`input_output_response` function, which handles
simulation of nonlinear systems and interconnected systems, can be
used.  For an LTI system, results are generally more accurate using
the LTI simulation functions above.  The :func:`input_output_response`
function is described in more detail in the :ref:`iosys-module` section.

.. _time-series-convention:

Time series data
----------------
A variety of functions in the library return time series data: sequences of
values that change over time.  A common set of conventions is used for
returning such data: columns represent different points in time, rows are
different components (e.g., inputs, outputs or states).  For return
arguments, an array of times is given as the first returned argument,
followed by one or more arrays of variable values.  This convention is used
throughout the library, for example in the functions
:func:`forced_response`, :func:`step_response`, :func:`impulse_response`,
and :func:`initial_response`.

.. note::
    The convention used by python-control is different from the convention
    used in the `scipy.signal
    <https://docs.scipy.org/doc/scipy/reference/signal.html>`_ library. In
    Scipy's convention the meaning of rows and columns is interchanged.
    Thus, all 2D values must be transposed when they are used with functions
    from `scipy.signal`_.

The time vector is a 1D array with shape (n, )::

      T = [t1,     t2,     t3,     ..., tn    ]

Input, state, and output all follow the same convention. Columns are different
points in time, rows are different components::

      U = [[u1(t1), u1(t2), u1(t3), ..., u1(tn)]
           [u2(t1), u2(t2), u2(t3), ..., u2(tn)]
           ...
           ...
           [ui(t1), ui(t2), ui(t3), ..., ui(tn)]]

(and similarly for `X`, `Y`).  So, `U[:, 2]` is the system's input at the
third point in time; and `U[1]` or `U[1, :]` is the sequence of values for
the system's second input.

When there is only one row, a 1D object is accepted or returned, which adds
convenience for SISO systems:

The initial conditions are either 1D, or 2D with shape (j, 1)::

     X0 = [[x1]
           [x2]
           ...
           ...
           [xj]]

Functions that return time responses (e.g., :func:`forced_response`,
:func:`impulse_response`, :func:`input_output_response`,
:func:`initial_response`, and :func:`step_response`) return a
:class:`TimeResponseData` object that contains the data for the time
response.  These data can be accessed via the
:attr:`~TimeResponseData.time`, :attr:`~TimeResponseData.outputs`,
:attr:`~TimeResponseData.states` and :attr:`~TimeResponseData.inputs`
properties::

    sys = ct.rss(4, 1, 1)
    response = ct.step_response(sys)
    plt.plot(response.time, response.outputs)

The dimensions of the response properties depend on the function being
called and whether the system is SISO or MIMO.  In addition, some time
response function can return multiple "traces" (input/output pairs),
such as the :func:`step_response` function applied to a MIMO system,
which will compute the step response for each input/output pair.  See
:class:`TimeResponseData` for more details.

The input, output, and state elements of the response can be accessed using
signal names in place of integer offsets::

    plt.plot(response. time, response.states['x[1]']

For multi-trace systems generated by :func:`step_response` and
:func:`impulse_response`, the input name used to generate the trace can be
used to access the appropriate input output pair::

    plt.plot(response.time, response.outputs['y[0]', 'u[1]'])

The time response functions can also be assigned to a tuple, which extracts
the time and output (and optionally the state, if the `return_x` keyword is
used).  This allows simple commands for plotting::

    t, y = ct.step_response(sys)
    plot(t, y)

The output of a MIMO LTI system can be plotted like this::

    t, y = ct.forced_response(sys, t, u)
    plot(t, y[0], label='y_0')
    plot(t, y[1], label='y_1')

The convention also works well with the state space form of linear
systems. If `D` is the feedthrough matrix (2D array) of a linear system,
and `U` is its input (array), then the feedthrough part of the system's
response, can be computed like this::

    ft = D @ U

Finally, the `to_pandas()` function can be used to create a pandas dataframe::

    df = response.to_pandas()

The column labels for the data frame are `time` and the labels for the input,
output, and state signals (`u[i]`, `y[i]`, and `x[i]` by default, but these
can be changed using the `inputs`, `outputs`, and `states` keywords when
constructing the system, as described in :func:`ss`, :func:`tf`, and other
system creation function.  Note that when exporting to pandas, "rows" in the
data frame correspond to time and "cols" (DataSeries) correspond to signals.

Time response plots
-------------------

.. todo:: Improve flow between previous section and this one

Input/output time responses are produced one of several python-control
functions: :func:`forced_response`,
:func:`impulse_response`, :func:`initial_response`,
:func:`input_output_response`, :func:`step_response`.
Each of these return a :class:`TimeResponseData` object, which
contains the time, input, state, and output vectors associated with the
simulation. Time response data can be plotted with the
:func:`time_response_plot` function, which is also available as
the :func:`TimeResponseData.plot` method.  For example, the step
response for a two-input, two-output can be plotted using the commands::

  sys_mimo = ct.tf2ss(
      [[[1], [0.1]], [[0.2], [1]]],
      [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="sys_mimo")
  response = ct.step_response(sys)
  response.plot()

which produces the following plot:

.. image:: figures/timeplot-mimo_step-default.png

The  :class:`TimeResponseData` object can also be used to access
the data from the simulation::

  time, outputs, inputs = response.time, response.outputs, response.inputs
  fig, axs = plt.subplots(2, 2)
  for i in range(2):
      for j in range(2):
          axs[i, j].plot(time, outputs[i, j])

In addition to accessing time response data via integer indices, signal
names can allow be used::

  plt.plot(response.time, response.outputs['y[0]', 'u[1]'])

A number of options are available in the `plot` method to customize
the appearance of input output data.  For data produced by the
:func:`impulse_response` and :func:`step_response`
commands, the inputs are not shown.  This behavior can be changed
using the `plot_inputs` keyword.  It is also possible to combine
multiple lines onto a single graph, using either the `overlay_signals`
keyword (which puts all outputs out a single graph and all inputs on a
single graph) or the `overlay_traces` keyword, which puts different
traces (e.g., corresponding to step inputs in different channels) on
the same graph, with appropriate labeling via a legend on selected
axes.

For example, using `plot_input=True` and `overlay_signals=True` yields the
following plot::

      ct.step_response(sys_mimo).plot(
        plot_inputs=True, overlay_signals=True,
        title="Step response for 2x2 MIMO system " +
        "[plot_inputs, overlay_signals]")

.. image:: figures/timeplot-mimo_step-pi_cs.png

Input/output response plots created with either the
:func:`forced_response` or the
:func:`input_output_response` functions include the input signals by
default. These can be plotted on separate axes, but also "overlaid" on the
output axes (useful when the input and output signals are being compared to
each other).  The following plot shows the use of `plot_inputs='overlay'`
as well as the ability to reposition the legends using the `legend_map`
keyword::

  timepts = np.linspace(0, 10, 100)
  U = np.vstack([np.sin(timepts), np.cos(2*timepts)])
  ct.input_output_response(sys_mimo, timepts, U).plot(
      plot_inputs='overlay',
      legend_map=np.array([['lower right'], ['lower right']]),
      title="I/O response for 2x2 MIMO system " +
      "[plot_inputs='overlay', legend_map]")

.. image:: figures/timeplot-mimo_ioresp-ov_lm.png

Another option that is available is to use the `transpose` keyword so that
instead of plotting the outputs on the top and inputs on the bottom, the
inputs are plotted on the left and outputs on the right, as shown in the
following figure::

  U1 = np.vstack([np.sin(timepts), np.cos(2*timepts)])
  resp1 = ct.input_output_response(sys_mimo, timepts, U1)

  U2 = np.vstack([np.cos(2*timepts), np.sin(timepts)])
  resp2 = ct.input_output_response(sys_mimo, timepts, U2)

  ct.combine_time_responses(
      [resp1, resp2], trace_labels=["Scenario #1", "Scenario #2"]).plot(
          transpose=True,
          title="I/O responses for 2x2 MIMO system, multiple traces "
          "[transpose]")

.. image:: figures/timeplot-mimo_ioresp-mt_tr.png

This figure also illustrates the ability to create "multi-trace" plots
using the :func:`combine_time_responses` function.  The line
properties that are used when combining signals and traces are set by
the `input_props`, `output_props` and `trace_props` parameters for
:func:`time_response_plot`.

Additional customization is possible using the `input_props`,
`output_props`, and `trace_props` keywords to set complementary line colors
and styles for various signals and traces::

  cplt = ct.step_response(sys_mimo).plot(
      plot_inputs='overlay', overlay_signals=True, overlay_traces=True,
      output_props=[{'color': c} for c in ['blue', 'orange']],
      input_props=[{'color': c} for c in ['red', 'green']],
      trace_props=[{'linestyle': s} for s in ['-', '--']])

.. image:: figures/timeplot-mimo_step-linestyle.png


.. _frequency_response:

Frequency response data
=======================

Linear time invariant (LTI) systems can be analyzed in terms of their
frequency response and python-control provides a variety of tools for
carrying out frequency response analysis.  The most basic of these is
the :func:`frequency_response` function, which will compute
the frequency response for one or more linear systems::

  sys1 = ct.tf([1], [1, 2, 1], name='sys1')
  sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
  response = ct.frequency_response([sys1, sys2])

A Bode plot provide a graphical view of the response an LTI system and can
be generated using the :func:`bode_plot` function::

  ct.bode_plot(response, initial_phase=0)

.. image:: figures/freqplot-siso_bode-default.png

Computing the response for multiple systems at the same time yields a
common frequency range that covers the features of all listed systems.

Bode plots can also be created directly using the
:meth:`FrequencyResponseData.plot` method::

  sys_mimo = ct.tf(
      [[[1], [0.1]], [[0.2], [1]]],
      [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="sys_mimo")
  ct.frequency_response(sys_mimo).plot()

.. image:: figures/freqplot-mimo_bode-default.png

A variety of options are available for customizing Bode plots, for
example allowing the display of the phase to be turned off or
overlaying the inputs or outputs::

  ct.frequency_response(sys_mimo).plot(
      plot_phase=False, overlay_inputs=True, overlay_outputs=True)

.. image:: figures/freqplot-mimo_bode-magonly.png

The :func:`singular_values_response` function can be used to
generate Bode plots that show the singular values of a transfer
function::

  ct.singular_values_response(sys_mimo).plot()

.. image:: figures/freqplot-mimo_svplot-default.png

Different types of plots can also be specified for a given frequency
response.  For example, to plot the frequency response using a a Nichols
plot, use `plot_type='nichols'`::

  response.plot(plot_type='nichols')

.. image:: figures/freqplot-siso_nichols-default.png

Another response function that can be used to generate Bode plots is the
:func:`gangof4_response` function, which computes the four primary
sensitivity functions for a feedback control system in standard form::

  proc = ct.tf([1], [1, 1, 1], name="process")
  ctrl = ct.tf([100], [1, 5], name="control")
  response = rect.gangof4_response(proc, ctrl)
  ct.bode_plot(response)	# or response.plot()

.. image:: figures/freqplot-gangof4.png

Nyquist analysis can be done using the :func:`nyquist_response`
function, which evaluates an LTI system along the Nyquist contour, and
the :func:`nyquist_plot` function, which generates a Nyquist plot::

  sys = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys')
  nyquist_plot(sys)

.. image:: figures/freqplot-nyquist-default.png

The :func:`nyquist_response` function can be used to compute
the number of encirclements of the -1 point and can return the Nyquist
contour that was used to generate the Nyquist curve.

By default, the Nyquist response will generate small semicircles around
poles that are on the imaginary axis.  In addition, portions of the Nyquist
curve that are far from the origin are scaled to a maximum value, while the
line style is changed to reflect the scaling, and it is possible to offset
the scaled portions to separate out the portions of the Nyquist curve at
:math:`\infty`.  A number of keyword parameters for both are available for
:func:`nyquist_response` and :func:`nyquist_plot` to tune
the computation of the Nyquist curve and the way the data are plotted::

  sys = ct.tf([1, 0.2], [1, 0, 1]) * ct.tf([1], [1, 0])
  nyqresp = ct.nyquist_response(sys)
  nyqresp.plot(
      max_curve_magnitude=6, max_curve_offset=1,
      arrows=[0, 0.15, 0.3, 0.6, 0.7, 0.925], label='sys')
  print("Encirclements =", nyqresp.count)

.. image:: figures/freqplot-nyquist-custom.png

All frequency domain plotting functions will automatically compute the
range of frequencies to plot based on the poles and zeros of the frequency
response.  Frequency points can be explicitly specified by including an
array of frequencies as a second argument (after the list of systems)::

  sys1 = ct.tf([1], [1, 2, 1], name='sys1')
  sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
  omega = np.logspace(-2, 2, 500)
  ct.frequency_response([sys1, sys2], omega).plot(initial_phase=0)

.. image:: figures/freqplot-siso_bode-omega.png

Alternatively, frequency ranges can be specified by passing a list of the
form `[wmin, wmax]`, where `wmin` and `wmax` are the minimum and
maximum frequencies in the (log-spaced) frequency range::

  response = ct.frequency_response([sys1, sys2], [1e-2, 1e2])

The number of (log-spaced) points in the frequency can be specified using
the `omega_num` keyword parameter.

Frequency response data can also be accessed directly and plotted manually::

  sys = ct.rss(4, 2, 2, strictly_proper=True)  # 2x2 MIMO system
  fresp = ct.frequency_response(sys)
  plt.loglog(fresp.omega, fresp.magnitude['y[1]', 'u[0]'])

Access to frequency response data is available via the attributes
`omega`, `magnitude`,` `phase`, and `response`, where `response`
represents the complex value of the frequency response at each frequency.
The `magnitude`, `phase`, and `response` arrays can be indexed using
either input/output indices or signal names, with the first index
corresponding to the output signal and the second input corresponding to
the input signal.

Pole/zero data
==============

Pole/zero maps and root locus diagrams provide insights into system
response based on the locations of system poles and zeros in the complex
plane.  The :func:`pole_zero_map` function returns the poles and
zeros and can be used to generate a pole/zero plot::

  sys = ct.tf([1, 2], [1, 2, 3], name='SISO transfer function')
  response = ct.pole_zero_map(sys)
  ct.pole_zero_plot(response)

.. image:: figures/pzmap-siso_ctime-default.png

A root locus plot shows the location of the closed loop poles of a system
as a function of the loop gain::

  ct.root_locus_map(sys).plot()

.. image:: figures/rlocus-siso_ctime-default.png

The grid in the left hand plane shows lines of constant damping ratio as
well as arcs corresponding to the frequency of the complex pole.  The grid
can be turned off using the `grid` keyword.  Setting `grid` to `False` will
turn off the grid but show the real and imaginary axis.  To completely
remove all lines except the root loci, use `grid='empty'`.

On systems that support interactive plots, clicking on a location on the
root locus diagram will mark the pole locations on all branches of the
diagram and display the gain and damping ratio for the clicked point below
the plot title:

.. image:: figures/rlocus-siso_ctime-clicked.png

Root locus diagrams are also supported for discrete time systems, in which
case the grid is show inside the unit circle::

  sysd = sys.sample(0.1)
  ct.root_locus_plot(sysd)

.. image:: figures/rlocus-siso_dtime-default.png

Lists of systems can also be given, in which case the root locus diagram
for each system is plotted in different colors::

  sys1 = ct.tf([1], [1, 2, 1], name='sys1')
  sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
  ct.root_locus_plot([sys1, sys2], grid=False)

.. image:: figures/rlocus-siso_multiple-nogrid.png


Customizing control plots
=========================

A set of common options are available to customize control plots in
various ways.  The following general rules apply:

* If a plotting function is called multiple times with data that generate
  control plots with the same shape for the array of subplots, the new data
  will be overlaid with the old data, with a change in color(s) for the
  new data (chosen from the standard matplotlib color cycle).  If not
  overridden, the plot title and legends will be updated to reflect all
  data shown on the plot.

* If a plotting function is called and the shape for the array of subplots
  does not match the currently displayed plot, a new figure is created.
  Note that only the shape is checked, so if two different types of
  plotting commands that generate the same shape of subplots are called
  sequentially, the :func:`matplotlib.pyplot.figure` command should be used
  to explicitly create a new figure.

* The `ax` keyword argument can be used to direct the plotting function
  to use a specific axes or array of axes.  The value of the `ax` keyword
  must have the proper number of axes for the plot (so a plot generating a
  2x2 array of subplots should be given a 2x2 array of axes for the `ax`
  keyword).

* The `color`, `linestyle`, `linewidth`, and other matplotlib line
  property arguments can be used to override the default line properties.
  If these arguments are absent, the default matplotlib line properties are
  used and the color cycles through the default matplotlib color cycle.

  The :func:`bode_plot`, :func:`time_response_plot`,
  and selected other commands can also accept a matplotlib format
  string (e.g., `'r--'`).  The format string must appear as a positional
  argument right after the required data argument.

  Note that line property arguments are the same for all lines generated as
  part of a single plotting command call, including when multiple responses
  are passed as a list to the plotting command.  For this reason it is
  often easiest to call multiple plot commands in sequence, with each
  command setting the line properties for that system/trace.

* The `label` keyword argument can be used to override the line labels
  that are used in generating the title and legend.  If more than one line
  is being plotted in a given call to a plot command, the `label`
  argument value should be a list of labels, one for each line, in the
  order they will appear in the legend.

  For input/output plots (frequency and time responses), the labels that
  appear in the legend are of the form "<output name>, <input name>, <trace
  name>, <system name>".  The trace name is used only for multi-trace time
  plots (for example, step responses for MIMO systems).  Common information
  present in all traces is removed, so that the labels appearing in the
  legend represent the unique characteristics of each line.

  For non-input/output plots (e.g., Nyquist plots, pole/zero plots, root
  locus plots), the default labels are the system name.

  If `label` is set to `False`, individual lines are still given
  labels, but no legend is generated in the plot. (This can also be
  accomplished by setting `legend_map` to `False`).

  Note: the `label` keyword argument is not implemented for describing
  function plots or phase plane plots, since these plots are primarily
  intended to be for a single system.  Standard `matplotlib` commands can
  be used to customize these plots for displaying information for multiple
  systems.

* The `legend_loc`, `legend_map` and `show_legend` keyword arguments
  can be used to customize the locations for legends.  By default, a
  minimal number of legends are used such that lines can be uniquely
  identified and no legend is generated if there is only one line in the
  plot.  Setting `show_legend` to `False` will suppress the legend and
  setting it to `True` will force the legend to be displayed even if
  there is only a single line in each axes.  In addition, if the value of
  the `legend_loc` keyword argument is set to a string or integer, it
  will set the position of the legend as described in the
  :func:`matplotlib.legend` documentation.  Finally, `legend_map` can be
  set to an array that matches the shape of the subplots, with each item
  being a string indicating the location of the legend for that axes (or
  `None` for no legend).

* The `rcParams` keyword argument can be used to override the default
  matplotlib style parameters used when creating a plot.  The default
  parameters for all control plots are given by the `ct.rcParams`
  dictionary and have the following values:

  .. list-table::
     :widths: 50 50
     :header-rows: 1

     * - Key
       - Value
     * - 'axes.labelsize'
       - 'small'
     * - 'axes.titlesize'
       - 'small'
     * - 'figure.titlesize'
       - 'medium'
     * - 'legend.fontsize'
       - 'x-small'
     * - 'xtick.labelsize'
       - 'small'
     * - 'ytick.labelsize'
       - 'small'

  Only those values that should be changed from the default need to be
  specified in the `rcParams` keyword argument.  To override the defaults
  for all control plots, update the `ct.rcParams` dictionary entries.

  The default values for style parameters for control plots can be restored
  using :func:`reset_rcParams`.

* For multi-input, multi-output time and frequency domain plots, the
  `sharex` and `sharey` keyword arguments can be used to determine whether
  and how axis limits are shared between the individual subplots.  Setting
  the keyword to 'row' will share the axes limits across all subplots in a
  row, 'col' will share across all subplots in a column, 'all' will share
  across all subplots in the figure, and `False` will allow independent
  limits for each subplot.

  For Bode plots, the `share_magnitude` and `share_phase` keyword arguments
  can be used to independently control axis limit sharing for the magnitude
  and phase portions of the plot, and `share_frequency` can be used instead
  of `sharex`.

* The `title` keyword can be used to override the automatic creation of
  the plot title.  The default title is a string of the form "<Type> plot
  for <syslist>" where <syslist> is a list of the sys names contained in
  the plot (which can be updated if the plotting is called multiple times).
  Use `title=False` to suppress the title completely.  The title can also
  be updated using the :func:`ControlPlot.set_plot_title` method
  for the returned control plot object.

  The plot title is only generated if `ax` is `None`.

The following code illustrates the use of some of these customization
features::

    P = ct.tf([0.02], [1, 0.1, 0.01])   # servomechanism
    C1 = ct.tf([1, 1], [1, 0])          # unstable
    L1 = P * C1
    C2 = ct.tf([1, 0.05], [1, 0])       # stable
    L2 = P * C2

    plt.rcParams.update(ct.rcParams)
    fig = plt.figure(figsize=[7, 4])
    ax_mag = fig.add_subplot(2, 2, 1)
    ax_phase = fig.add_subplot(2, 2, 3)
    ax_nyquist = fig.add_subplot(1, 2, 2)

    ct.bode_plot(
        [L1, L2], ax=[ax_mag, ax_phase],
        label=["$L_1$ (unstable)", "$L_2$ (unstable)"],
        show_legend=False)
    ax_mag.set_title("Bode plot for $L_1$, $L_2$")
    ax_mag.tick_params(labelbottom=False)
    fig.align_labels()

    ct.nyquist_plot(L1, ax=ax_nyquist, label="$L_1$ (unstable)")
    ct.nyquist_plot(
        L2, ax=ax_nyquist, label="$L_2$ (stable)",
        max_curve_magnitude=22, legend_loc='upper right')
    ax_nyquist.set_title("Nyquist plot for $L_1$, $L_2$")

    fig.suptitle("Loop analysis for servomechanism control design")
    plt.tight_layout()

.. image:: figures/ctrlplot-servomech.png

As this example illustrates, python-control plotting functions and
Matplotlib plotting functions can generally be intermixed.  One type of
plot for which this does not currently work is pole/zero plots with a
continuous time omega-damping grid (including root locus diagrams), due to
the way that axes grids are implemented.  As a workaround, the
:func:`pole_zero_subplots` command can be used to create an array
of subplots with different grid types, as illustrated in the following
example::

    ax_array = ct.pole_zero_subplots(2, 1, grid=[True, False])
    sys1 = ct.tf([1, 2], [1, 2, 3], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    ct.root_locus_plot([sys1, sys2], ax=ax_array[0, 0])
    cplt = ct.root_locus_plot([sys1, sys2], ax=ax_array[1, 0])
    cplt.set_plot_title("Root locus plots (w/ specified axes)")

.. image:: figures/ctrlplot-pole_zero_subplots.png

Alternatively, turning off the omega-damping grid (using `grid=False` or
`grid='empty'`) allows use of Matplotlib layout commands.


Response and plotting functions
===============================

Response functions
------------------

Response functions take a system or list of systems and return a response
object that can be used to retrieve information about the system (e.g., the
number of encirclements for a Nyquist plot) as well as plotting (via the
`plot` method).

.. autosummary::

   describing_function_response
   frequency_response
   forced_response
   gangof4_response
   impulse_response
   initial_response
   input_output_response
   nyquist_response
   pole_zero_map
   root_locus_map
   singular_values_response
   step_response

Plotting functions
------------------

.. autosummary::

   bode_plot
   describing_function_plot
   nichols_plot
   nyquist_plot
   phase_plane_plot
   phaseplot.circlegrid
   phaseplot.equilpoints
   phaseplot.meshgrid
   phaseplot.separatrices
   phaseplot.streamlines
   phaseplot.vectorfield
   pole_zero_plot
   root_locus_plot
   singular_values_plot
   time_response_plot


Utility functions
-----------------
These additional functions can be used to manipulate response data or
carry out other operations in creating control plots.


.. autosummary::

   phaseplot.boxgrid
   combine_time_responses
   pole_zero_subplots
   reset_rcParams


Response and plotting classes
-----------------------------

The following classes are used in generating response data.

.. autosummary::

   ControlPlot
   DescribingFunctionResponse
   FrequencyResponseData
   FrequencyResponseList
   NyquistResponseData
   PoleZeroData
   TimeResponseData
   TimeResponseList
