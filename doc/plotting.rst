.. _plotting-module:

*************
Plotting data
*************

The Python Control Systems Toolbox contains a number of functions for
plotting input/output responses in the time and frequency domain, root
locus diagrams, and other standard charts used in control system analysis,
for example::

  bode_plot(sys)
  nyquist_plot([sys1, sys2])
  pole_zero_plot(sys)
  root_locus_plot(sys)

While plotting functions can be called directly, the standard pattern used
in the toolbox is to provide a function that performs the basic computation
or analysis (e.g., computation of the time or frequency response) and
returns and object representing the output data.  A separate plotting
function, typically ending in `_plot` is then used to plot the data,
resulting in the following standard pattern::

  response = nyquist_response([sys1, sys2])
  count = response.count          # number of encirclements of -1
  lines = nyquist_plot(response)  # Nyquist plot

The returned value `lines` provides access to the individual lines in the
generated plot, allowing various aspects of the plot to be modified to suit
specific needs.

The plotting function is also available via the `plot()` method of the
analysis object, allowing the following type of calls::

  step_response(sys).plot()
  frequency_response(sys).plot()
  nyquist_response(sys).plot()
  root_locus_map(sys).plot()

The remainder of this chapter provides additional documentation on how
these response and plotting functions can be customized.


Time response data
==================

Input/output time responses are produced one of several python-control
functions: :func:`~control.forced_response`,
:func:`~control.impulse_response`, :func:`~control.initial_response`,
:func:`~control.input_output_response`, :func:`~control.step_response`.
Each of these return a :class:`~control.TimeResponseData` object, which
contains the time, input, state, and output vectors associated with the
simulation. Time response data can be plotted with the
:func:`~control.time_response_plot` function, which is also available as
the :func:`~control.TimeResponseData.plot` method.  For example, the step
response for a two-input, two-output can be plotted using the commands::

  sys_mimo = ct.tf2ss(
      [[[1], [0.1]], [[0.2], [1]]],
      [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="sys_mimo")
  response = step_response(sys)
  response.plot()

which produces the following plot:

.. image:: timeplot-mimo_step-default.png

The  :class:`~control.TimeResponseData` object can also be used to access
the data from the simulation::

  time, outputs, inputs = response.time, response.outputs, response.inputs
  fig, axs = plt.subplots(2, 2)
  for i in range(2):
      for j in range(2):
          axs[i, j].plot(time, outputs[i, j])

A number of options are available in the `plot` method to customize
the appearance of input output data.  For data produced by the
:func:`~control.impulse_response` and :func:`~control.step_response`
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

.. image:: timeplot-mimo_step-pi_cs.png

Input/output response plots created with either the
:func:`~control.forced_response` or the
:func:`~control.input_output_response` functions include the input signals by
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

.. image:: timeplot-mimo_ioresp-ov_lm.png

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

.. image:: timeplot-mimo_ioresp-mt_tr.png

This figure also illustrates the ability to create "multi-trace" plots
using the :func:`~control.combine_time_responses` function.  The line
properties that are used when combining signals and traces are set by
the `input_props`, `output_props` and `trace_props` parameters for
:func:`~control.time_response_plot`.

Additional customization is possible using the `input_props`,
`output_props`, and `trace_props` keywords to set complementary line colors
and styles for various signals and traces::

    out = ct.step_response(sys_mimo).plot(
        plot_inputs='overlay', overlay_signals=True, overlay_traces=True,
        output_props=[{'color': c} for c in ['blue', 'orange']],
        input_props=[{'color': c} for c in ['red', 'green']],
        trace_props=[{'linestyle': s} for s in ['-', '--']])

.. image:: timeplot-mimo_step-linestyle.png

Frequency response data
=======================

Linear time invariant (LTI) systems can be analyzed in terms of their
frequency response and python-control provides a variety of tools for
carrying out frequency response analysis.  The most basic of these is
the :func:`~control.frequency_response` function, which will compute
the frequency response for one or more linear systems::

  sys1 = ct.tf([1], [1, 2, 1], name='sys1')
  sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
  response = ct.frequency_response([sys1, sys2])

A Bode plot provide a graphical view of the response an LTI system and can
be generated using the :func:`~control.bode_plot` function::

  ct.bode_plot(response, initial_phase=0)

.. image:: freqplot-siso_bode-default.png

Computing the response for multiple systems at the same time yields a
common frequency range that covers the features of all listed systems.

Bode plots can also be created directly using the
:meth:`~control.FrequencyResponseData.plot` method::

  sys_mimo = ct.tf(
      [[[1], [0.1]], [[0.2], [1]]],
      [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="sys_mimo")
  ct.frequency_response(sys_mimo).plot()

.. image:: freqplot-mimo_bode-default.png

A variety of options are available for customizing Bode plots, for
example allowing the display of the phase to be turned off or
overlaying the inputs or outputs::

  ct.frequency_response(sys_mimo).plot(
      plot_phase=False, overlay_inputs=True, overlay_outputs=True)

.. image:: freqplot-mimo_bode-magonly.png

The :func:`~ct.singular_values_response` function can be used to
generate Bode plots that show the singular values of a transfer
function::

  ct.singular_values_response(sys_mimo).plot()

.. image:: freqplot-mimo_svplot-default.png

Different types of plots can also be specified for a given frequency
response.  For example, to plot the frequency response using a a Nichols
plot, use `plot_type='nichols'`::

  response.plot(plot_type='nichols')

.. image:: freqplot-siso_nichols-default.png

Another response function that can be used to generate Bode plots is
the :func:`~ct.gangof4` function, which computes the four primary
sensitivity functions for a feedback control system in standard form::

    proc = ct.tf([1], [1, 1, 1], name="process")
    ctrl = ct.tf([100], [1, 5], name="control")
    response = rect.gangof4_response(proc, ctrl)
    ct.bode_plot(response)	# or response.plot()

.. image:: freqplot-gangof4.png


Pole/zero data
==============

Pole/zero maps and root locus diagrams provide insights into system
response based on the locations of system poles and zeros in the complex
plane.  The :func:`~control.pole_zero_map` function returns the poles and
zeros and can be used to generate a pole/zero plot::

  sys = ct.tf([1, 2], [1, 2, 3], name='SISO transfer function')
  response = ct.pole_zero_map(sys)
  ct.pole_zero_plot(response)

.. image:: pzmap-siso_ctime-default.png

A root locus plot shows the location of the closed loop poles of a system
as a function of the loop gain::

  ct.root_locus_map(sys).plot()

.. image:: rlocus-siso_ctime-default.png

The grid in the left hand plane shows lines of constant damping ratio as
well as arcs corresponding to the frequency of the complex pole.  The grid
can be turned off using the `grid` keyword.  Setting `grid` to `False` will
turn off the grid but show the real and imaginary axis.  To completely
remove all lines except the root loci, use `grid='empty'`.

On systems that support interactive plots, clicking on a location on the
root locus diagram will mark the pole locations on all branches of the
diagram and display the gain and damping ratio for the clicked point below
the plot title:

.. image:: rlocus-siso_ctime-clicked.png

Root locus diagrams are also supported for discrete time systems, in which
case the grid is show inside the unit circle::

  sysd = sys.sample(0.1)
  ct.root_locus_plot(sysd)

.. image:: rlocus-siso_dtime-default.png

Lists of systems can also be given, in which case the root locus diagram
for each system is plotted in different colors::

  sys1 = ct.tf([1], [1, 2, 1], name='sys1')
  sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
  ct.root_locus_plot([sys1, sys2], grid=False)

.. image:: rlocus-siso_multiple-nogrid.png


Response and plotting functions
===============================

Response functions
------------------

Response functions take a system or list of systems and return a response
object that can be used to retrieve information about the system (e.g., the
number of encirclements for a Nyquist plot) as well as plotting (via the
`plot` method).

.. autosummary::
   :toctree: generated/

   ~control.describing_function_response
   ~control.frequency_response
   ~control.forced_response
   ~control.gangof4_response
   ~control.impulse_response
   ~control.initial_response
   ~control.input_output_response
   ~control.nyquist_response
   ~control.pole_zero_map
   ~control.root_locus_map
   ~control.singular_values_response
   ~control.step_response

Plotting functions
------------------

.. autosummary::
   :toctree: generated/

   ~control.bode_plot
   ~control.describing_function_plot
   ~control.nichols_plot
   ~control.pole_zero_plot
   ~control.root_locus_plot
   ~control.singular_values_plot
   ~control.time_response_plot


Utility functions
-----------------

These additional functions can be used to manipulate response data or
returned values from plotting routines.

.. autosummary::
   :toctree: generated/

   ~control.combine_time_responses
   ~control.get_plot_axes


Response classes
----------------

The following classes are used in generating response data.

.. autosummary::
   :toctree: generated/

   ~control.DescribingFunctionResponse
   ~control.FrequencyResponseData
   ~control.NyquistResponseData
   ~control.PoleZeroData
   ~control.TimeResponseData
