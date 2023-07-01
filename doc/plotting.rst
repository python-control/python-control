.. _plotting-module:

*************
Plotting data
*************

The Python Control Toolbox contains a number of functions for plotting
input/output responses in the time and frequency domain, root locus
diagrams, and other standard charts used in control system analysis.  While
some legacy functions do both analysis and plotting, the standard pattern
used in the toolbox is to provide a function that performs the basic
computation (e.g., time or frequency response) and returns and object
representing the output data.  A separate plotting function, typically
ending in `_plot` is then used to plot the data.  The plotting function is
also available via the `plot()` method of the analysis object, allowing the
following type of calls::

  step_response(sys).plot()
  frequency_response(sys).plot()  # implementation pending
  nyquist_curve(sys).plot()       # implementation pending
  rootlocus_curve(sys).plot()     # implementation pending

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
      [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
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
:func:`~control.input_output_response` include the input signals by
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

    ct.combine_traces(
        [resp1, resp2], trace_labels=["Scenario #1", "Scenario #2"]).plot(
            transpose=True,
            title="I/O responses for 2x2 MIMO system, multiple traces "
            "[transpose]")
	    
.. image:: timeplot-mimo_ioresp-mt_tr.png

This figure also illustrates the ability to create "multi-trace" plots
using the :func:`~control.combine_traces` function.  The line properties
that are used when combining signals and traces are set by the
`input_props`, `output_props` and `trace_props` parameters for
:func:`~control.time_response_plot`.

Plotting functions
==================

.. autosummary::
   :toctree: generated/

   ~control.time_response_plot
   ~control.combine_traces
   ~control.get_plot_axes
