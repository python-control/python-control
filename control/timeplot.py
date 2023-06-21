# timeplot.py - time plotting functions
# RMM, 20 Jun 2023
#
# This file contains routines for plotting out time responses.  These
# functions can be called either as standalone functions or access from the
# TimeDataResponse class.
#
# Note: Depending on how this goes, it might eventually make sense to
# put the functions here directly into timeresp.py.
#
# Desired features
# [ ] Step/impulse response plots don't include inputs by default
# [ ] Forced/I-O response plots include inputs by default
# [ ] Ability to start inputs at zero
# [ ] Ability to plot all data on a single graph
# [ ] Ability to plot inputs with outputs on separate graphs
# [ ] Ability to plot inputs and/or outputs on selected axes
# [ ] Multi-trace graphs using different line styles
# [ ] Plotting function return Line2D elements
# [ ] Axis labels/legends based on what is plotted (siso, mimo, multi-trace)
# [ ] Ability to select (index) output and/or trace (and time?)
# [ ] Legends should not contain redundant information (nor appear redundantly)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# Default font dictionary
_timeplot_rcParams = mpl.rcParams.copy()
_timeplot_rcParams.update({
    'axes.labelsize': 'small',
    'axes.titlesize': 'small',
    'figure.titlesize': 'medium',
    'legend.fontsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
})

# Default values for module parameter variables
_timeplot_defaults = {
    'timeplot.line_styles': ['-', '--', ':', '-.'],
    'timeplot.line_colors': [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
    'timeplot.time_label': "Time [s]",
}

# Plot the input/output response of a system
def ioresp_plot(
        data, ax=None, plot_inputs=None, plot_outputs=True, transpose=False,
        combine_traces=False, combine_signals=False, legend_loc='center right',
        add_initial_zero=True, title=None, relabel=True, **kwargs):
    """Plot the time response of an input/output system.

    This function creates a standard set of plots for the input/output
    response of a system, with the data provided via a `TimeResponseData`
    object, which is the standard output for python-control simulation
    functions.

    Parameters
    ----------
    data : TimeResponseData
        Data to be plotted.
    ax : array of Axes
        The matplotlib Axes to draw the figure on.  If not specified, the
        Axes for the current figure are used or, if there is no current
        figure with the correct number and shape of Axes, a new figure is
        created.  The default shape of the array should be (data.ntraces,
        data.ninputs + data.inputs), but if combine_traces == True then
        only one row is needed and if combine_signals == True then only one
        or two columns are needed (depending on plot_inputs and
        plot_outputs).
    plot_inputs : str or bool, optional
        Sets how and where to plot the inputs:
            * False: don't plot the inputs.
            * 'separate` or True: plot inputs on their own axes
            * 'with-output': plot inputs on corresponding output axes
            * 'combined`: combine inputs and show on each output
    plot_outputs : bool, optional
        If False, suppress plotting of the outputs.
    combine_traces : bool, optional
        If set to True, combine all traces onto a single row instead of
        plotting a separate row for each trace.
    combine_signals : bool, optional
        If set to True, combine all input and output signals onto a single
        plot (for each).
    transpose : bool, optional
        If transpose is False (default), signals are plotted from top to
        bottom, starting with outputs (if plotted) and then inputs.
        Multi-trace plots are stacked horizontally.  If transpose is True,
        signals are plotted from left to right, starting with the inputs
        (if plotted) and then the outputs.  Multi-trace responses are
        stacked vertically.

    Returns
    -------
    out : list of Artist or list of list of Artist
        Array of Artist objects for each line in the plot.  The shape of
        the array matches the plot style,

    Additional Parameters
    ---------------------
    relabel : bool, optional
        By default, existing figures and axes are relabeled when new data
        are added.  If set to `False`, just plot new data on existing axes.
    time_label : str, optional
        Label to use for the time axis.
    legend_loc : str or list of str, optional
        Location of the legend for multi-trace plots.  If an array line
        style is used, a list of locations can be passed to allow
        specifying individual locations for the legend in each axes.
    add_initial_zero : bool
        Add an initial point of zero at the first time point for all
        inputs.  This is useful when the initial value of the input is
        nonzero (for example in a step input).  Default is True.
    trace_cycler: :class:`~matplotlib.Cycler`
        Line style cycle to use for traces.  Default = ['-', '--', ':', '-.'].

    """
    from cycler import cycler
    from .iosys import InputOutputSystem
    from .timeresp import TimeResponseData

    #
    # Process keywords and set defaults
    #

    # Set up defaults
    time_label = config._get_param(
        'timeplot', 'time_label', kwargs, _timeplot_defaults, pop=True)
    line_styles = config._get_param(
        'timeplot', 'line_styles', kwargs, _timeplot_defaults, pop=True)
    line_colors = config._get_param(
        'timeplot', 'line_colors', kwargs, _timeplot_defaults, pop=True)

    title = data.title if title == None else title

    # Make sure we process alled of the optional arguments
    if kwargs:
        raise TypeError("unrecognized keywords: " + str(kwargs))

    # Configure the cycle of colors and line styles
    my_cycler = cycler(linestyle=line_styles) * cycler(color=line_colors)

    #
    # Find/create axes
    #
    # Data are plotted in a standard subplots array, whose size depends on
    # which signals are being plotted and how they are combined.  The
    # baseline layout for data is to plot everything separately, with
    # outputs and inputs making up the rows and traces making up the
    # columns:
    #
    # Trace 0        Trace q
    # +------+       +------+
    # | y[0] |  ...  | y[0] |
    # +------+       +------+
    #    :
    # +------+       +------+
    # | y[p] |  ...  | y[p] |
    # +------+       +------+
    #
    # +------+       +------+
    # | u[0] |  ...  | u[0] |
    # +------+       +------+
    #    :
    # +------+       +------+
    # | u[m] |  ...  | u[m] |
    # +------+       +------+
    #
    # * Omitting: either the inputs or the outputs can be omitted.
    #
    # * Combining: inputs, outputs, and traces can be combined onto a
    #   single set of axes using various keyword combinations
    #   (combine_signals, combine_traces, plot_input='combine').  This
    #   basically collapses data along either the rows or columns, and a
    #   legend is generated.
    #
    # * Transpose: if the `transpose` keyword is True, then instead of
    #   plotting the data vertically (outputs over inputs), we plot left to
    #   right (inputs, outputs):
    #
    #         +------+       +------+     +------+       +------+
    # Trace 0 | u[0] |  ...  | u[m] |     | y[0] |  ...  | y[p] |
    #         +------+       +------+     +------+       +------+
    #    :
    #    :
    #         +------+       +------+     +------+       +------+
    # Trace q | u[0] |  ...  | u[m] |     | y[0] |  ...  | y[p] |
    #         +------+       +------+     +------+       +------+
    #

    # Decide on the number of inputs and outputs
    if plot_inputs is None:
        plot_inputs = data.plot_inputs
    ninputs = data.ninputs if plot_inputs else 0
    noutputs = data.noutputs if plot_outputs else 0
    if ninputs == 0 and noutputs == 0:
        raise ValueError(
            "plot_inputs and plot_outputs both True; no data to plot")

    # Figure how how many rows and columns to use
    nrows = noutputs + ninputs if not combine_signals else \
        int(plot_inputs) + int(plot_outputs)
    ncols = data.ntraces if not combine_traces else 1
    if transpose:
        nrows, ncols = ncols, nrows

    # See if we can use the current figure axes
    fig = plt.gcf()         # get current figure (or create new one)
    if ax is None and plt.get_fignums():
        ax = fig.get_axes()
        if len(ax) == nrows * ncols:
            # Assume that the shape is right (no easy way to infer this)
            ax = np.array(ax).reshape(nrows, ncols)
        elif len(ax) != 0:
            # Need to generate a new figure
            fig, ax = plt.figure(), None
        else:
            # Blank figure, just need to recreate axes
            ax = None

    # Create new axes, if needed, and customize them
    if ax is None:
        with plt.rc_context(_timeplot_rcParams):
            ax_array = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
            for ax in np.nditer(ax_array, flags=["refs_ok"]):
                ax.item().set_prop_cycle(my_cycler)
        fig.set_tight_layout(True)
        fig.align_labels()

    else:
        # Make sure the axes are the right shape
        if ax.shape != (nrows, ncols):
            raise ValueError(
                "specified axes are not the right shape; "
                f"got {ax.shape} but expecting ({nrows}, {ncols})")
        ax_array = ax

    #
    # Map inputs/outputs and traces to axes
    #
    # This set of code takes care of all of the various options for how to
    # plot the data.  The arrays ax_outputs and ax_inputs are used to map
    # the different signals that are plotted onto the axes created above.
    # This code is complicated because it has to handle lots of different
    # variations.
    #

    # Create the map from trace, signal to axes, accounting for combine_*
    ax_outputs = np.empty((noutputs, data.ntraces), dtype=object)
    ax_inputs = np.empty((ninputs, data.ntraces), dtype=object)

    # Keep track of the number of axes for the inputs and outputs
    noutput_axes = noutputs if plot_outputs and not combine_signals \
        else int(plot_outputs)
    ninput_axes = ninputs if plot_inputs and not combine_signals \
        else int(plot_inputs)

    for i in range(noutputs):
        for j in range(data.ntraces):
            signal_index = i if not combine_signals else 0
            trace_index = j if not combine_traces else 0
            if transpose:
                ax_outputs[i, j] = \
                    ax_array[trace_index, signal_index + ninput_axes]
            else:
                ax_outputs[i, j] = ax_array[signal_index, trace_index]

    for i in range(ninputs):
        for j in range(data.ntraces):
            signal_index = noutput_axes + (i if not combine_signals else 0)
            trace_index = j if not combine_traces else 0
            if transpose:
                ax_inputs[i, j] = \
                    ax_array[trace_index, signal_index - noutput_axes]
            else:
                ax_inputs[i, j] = ax_array[signal_index, trace_index]

    #
    # Plot the data
    #

    # Reshape the inputs and outputs for uniform indexing
    outputs = data.y.reshape(data.noutputs, data.ntraces, -1)
    inputs = data.u.reshape(data.ninputs, data.ntraces, -1)

    # Create a list of lines for the output
    out = np.empty((noutputs + ninputs, data.ntraces), dtype=object)

    # Go through each trace and each input/output
    for trace in range(data.ntraces):
        # Set the line style for each trace
        style = line_styles[trace % len(line_styles)]

        # Plot the output
        for i in range(noutputs):
            label = data.output_labels[i]
            if data.ntraces > 1:
                label += f", trace {trace}"
            out[i, trace] = ax_outputs[i, trace].plot(
                data.time, outputs[i][trace], label=label)

        # Plot the input
        for i in range(ninputs):
            label = data.input_labels[i]    # set label for legend
            if data.ntraces > 1:
                label += f", trace {trace}"

            if add_initial_zero:            # start trace from the origin
                x = np.hstack([np.array([data.time[0]]), data.time])
                y = np.hstack([np.array([0]), inputs[i][trace]])
            else:
                x, y = data.time, inputs[i][trace]

            out[noutputs + i, trace] = ax_inputs[i, trace].plot(
                x, y, label=label)

    # Stop here if the user wants to control everything
    if not relabel:
        return out

    #
    # Label the axes
    #

    # Label the outputs
    if combine_signals and plot_outputs:
        if transpose:
            for trace in range(data.ntraces if transpose else 1):
                ax_outputs[0, trace].set_ylabel("Outputs")
        else:
            ax_array[0, 0].set_ylabel("Outputs")
    else:
        for i in range(noutputs):
            for trace in range(data.ntraces if transpose else 1):
                ax_outputs[i, trace].set_ylabel(data.output_labels[i])

    # Label the inputs
    if combine_signals and plot_inputs:
        if transpose:
            for trace in range(data.ntraces if transpose else 1):
                ax_inputs[0, trace].set_ylabel("Inputs")
        else:
            ax_inputs[0, 0].set_ylabel("Inputs")
    else:
        for i in range(ninputs):
            for trace in range(data.ntraces if transpose else 1):
                ax_inputs[i, trace].set_ylabel(data.input_labels[i])

    # Label the traces
    if not combine_traces and not transpose:
        for trace in range(data.ntraces):
            with plt.rc_context(_timeplot_rcParams):
                ax_outputs[0, trace].set_title(f"Trace {trace}")

    # Create legends
    legend_loc = np.broadcast_to(np.array(legend_loc), ax_array.shape)
    for i in range(nrows):
        for j in range(ncols):
            ax = ax_array[i, j]
            if len(ax.get_lines()) > 1:
                with plt.rc_context(_timeplot_rcParams):
                    ax.legend(loc=legend_loc[i, j])

    # if data.noutputs > 1 or data.ntraces > 1:
    #     ax[0].set_ylabel("Outputs")
    #     ax[0].legend(loc=legend_loc[0])
    # else:
    #     ax[0].set_ylabel(f"Output {data.output_labels[i]}")

    # Time units on the bottom
    for col in range(ncols):
        ax_array[-1, col].set_xlabel(time_label)

    if fig is not None and data.title is not None:
        with plt.rc_context(_timeplot_rcParams):
            fig.suptitle(title)

    return out
