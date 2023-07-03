# timeplot.py - time plotting functions
# RMM, 20 Jun 2023
#
# This file contains routines for plotting out time responses.  These
# functions can be called either as standalone functions or access from the
# TimeDataResponse class.
#
# Note: It might eventually make sense to put the functions here
# directly into timeresp.py.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import commonprefix
from warnings import warn

from . import config

__all__ = ['time_response_plot', 'combine_traces', 'get_plot_axes']

# Default font dictionary
_timeplot_rcParams = mpl.rcParams.copy()
_timeplot_rcParams.update({
    'axes.labelsize': 'small',
    'axes.titlesize': 'small',
    'figure.titlesize': 'medium',
    'legend.fontsize': 'x-small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
})

# Default values for module parameter variables
_timeplot_defaults = {
    'timeplot.rcParams': _timeplot_rcParams,
    'timeplot.trace_props': [
        {'linestyle': s} for s in ['-', '--', ':', '-.']],
    'timeplot.output_props': [
        {'color': c} for c in [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:gray']],
    'timeplot.input_props': [
        {'color': c} for c in [
            'tab:red', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan']],
    'timeplot.time_label': "Time [s]",
}

# Plot the input/output response of a system
def time_response_plot(
        data, *fmt, ax=None, plot_inputs=None, plot_outputs=True,
        transpose=False, overlay_traces=False, overlay_signals=False,
        legend_map=None, legend_loc=None, add_initial_zero=True,
        trace_labels=None, title=None, relabel=True, **kwargs):
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
        created.  The default shape of the array should be (noutputs +
        ninputs, ntraces), but if `overlay_traces` is set to `True` then
        only one row is needed and if `overlay_signals` is set to `True`
        then only one or two columns are needed (depending on plot_inputs
        and plot_outputs).
    plot_inputs : bool or str, optional
        Sets how and where to plot the inputs:
            * False: don't plot the inputs
            * None: use value from time response data (default)
            * 'overlay`: plot inputs overlaid with outputs
            * True: plot the inputs on their own axes
    plot_outputs : bool, optional
        If False, suppress plotting of the outputs.
    overlay_traces : bool, optional
        If set to True, combine all traces onto a single row instead of
        plotting a separate row for each trace.
    overlay_signals : bool, optional
        If set to True, combine all input and output signals onto a single
        plot (for each).
    transpose : bool, optional
        If transpose is False (default), signals are plotted from top to
        bottom, starting with outputs (if plotted) and then inputs.
        Multi-trace plots are stacked horizontally.  If transpose is True,
        signals are plotted from left to right, starting with the inputs
        (if plotted) and then the outputs.  Multi-trace responses are
        stacked vertically.
    *fmt : :func:`matplotlib.pyplot.plot` format string, optional
        Passed to `matplotlib` as the format string for all lines in the plot.
    **kwargs : :func:`matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords passed to `matplotlib` to specify line properties.

    Returns
    -------
    out : array of list of Line2D
        Array of Line2D objects for each line in the plot.  The shape of
        the array matches the subplots shape and the value of the array is a
        list of Line2D objects in that subplot.

    Additional Parameters
    ---------------------
    add_initial_zero : bool
        Add an initial point of zero at the first time point for all
        inputs with type 'step'.  Default is True.
    input_props : array of dicts
        List of line properties to use when plotting combined inputs.  The
        default values are set by config.defaults['timeplot.input_props'].
    legend_map : array of str, option
        Location of the legend for multi-trace plots.  Specifies an array
        of legend location strings matching the shape of the subplots, with
        each entry being either None (for no legend) or a legend location
        string (see :func:`~matplotlib.pyplot.legend`).
    legend_loc : str
        Location of the legend within the axes for which it appears.  This
        value is used if legend_map is None.
    output_props : array of dicts
        List of line properties to use when plotting combined outputs.  The
        default values are set by config.defaults['timeplot.output_props'].
    relabel : bool, optional
        By default, existing figures and axes are relabeled when new data
        are added.  If set to `False`, just plot new data on existing axes.
    time_label : str, optional
        Label to use for the time axis.
    trace_props : array of dicts
        List of line properties to use when plotting combined outputs.  The
        default values are set by config.defaults['timeplot.trace_props'].

    Notes
    -----
    1. A new figure will be generated if there is no current figure or
       the current figure has an incompatible number of axes.  To
       force the creation of a new figures, use `plt.figure()`.  To reuse
       a portion of an existing figure, use the `ax` keyword.

    2. The line properties (color, linestyle, etc) can be set for the
       entire plot using the `fmt` and/or `kwargs` parameter, which
       are passed on to `matplotlib`.  When combining signals or
       traces, the `input_props`, `output_props`, and `trace_props`
       parameters can be used to pass a list of dictionaries
       containing the line properties to use.  These input/output
       properties are combined with the trace properties and finally
       the kwarg properties to determine the final line properties.

    3. The default plot properties, such as font sizes, can be set using
       config.defaults[''timeplot.rcParams'].

    """
    from .iosys import InputOutputSystem
    from .timeresp import TimeResponseData

    #
    # Process keywords and set defaults
    #

    # Set up defaults
    time_label = config._get_param(
        'timeplot', 'time_label', kwargs, _timeplot_defaults, pop=True)
    timeplot_rcParams = config._get_param(
        'timeplot', 'rcParams', kwargs, _timeplot_defaults, pop=True)

    if kwargs.get('input_props', None) and len(fmt) > 0:
        warn("input_props ignored since fmt string was present")
    input_props = config._get_param(
        'timeplot', 'input_props', kwargs, _timeplot_defaults, pop=True)
    iprop_len = len(input_props)

    if kwargs.get('output_props', None) and len(fmt) > 0:
        warn("output_props ignored since fmt string was present")
    output_props = config._get_param(
        'timeplot', 'output_props', kwargs, _timeplot_defaults, pop=True)
    oprop_len = len(output_props)

    if kwargs.get('trace_props', None) and len(fmt) > 0:
        warn("trace_props ignored since fmt string was present")
    trace_props = config._get_param(
        'timeplot', 'trace_props', kwargs, _timeplot_defaults, pop=True)
    tprop_len = len(trace_props)

    # Set the title for the data
    title = data.title if title == None else title

    # Determine whether or not to plot the input data (and how)
    if plot_inputs is None:
        plot_inputs = data.plot_inputs
    if plot_inputs not in [True, False, 'overlay']:
        raise ValueError(f"unrecognized value: {plot_inputs=}")

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
    # A variety of options are available to modify this format:
    #
    # * Omitting: either the inputs or the outputs can be omitted.
    #
    # * Combining: inputs, outputs, and traces can be combined onto a
    #   single set of axes using various keyword combinations
    #   (overlay_signals, overlay_traces, plot_inputs='overlay').  This
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
    # This also affects the way in which legends and labels are generated.

    # Decide on the number of inputs and outputs
    ninputs = data.ninputs if plot_inputs else 0
    noutputs = data.noutputs if plot_outputs else 0
    ntraces = max(1, data.ntraces)      # treat data.ntraces == 0 as 1 trace
    if ninputs == 0 and noutputs == 0:
        raise ValueError(
            "plot_inputs and plot_outputs both False; no data to plot")
    elif plot_inputs == 'overlay' and noutputs == 0:
        raise ValueError(
            "can't overlay inputs with no outputs")
    elif plot_inputs in [True, 'overlay'] and data.ninputs == 0:
        raise ValueError(
            "input plotting requested but no inputs in time response data")

    # Figure how how many rows and columns to use + offsets for inputs/outputs
    if plot_inputs == 'overlay' and not overlay_signals:
        nrows = max(ninputs, noutputs)          # Plot inputs on top of outputs
        noutput_axes = 0                        # No offset required
        ninput_axes = 0                         # No offset required
    elif overlay_signals:
        nrows = int(plot_outputs)               # Start with outputs
        nrows += int(plot_inputs == True)       # Add plot for inputs if needed
        noutput_axes = 1 if plot_outputs and plot_inputs is True else 0
        ninput_axes = 1 if plot_inputs is True else 0
    else:
        nrows = noutputs + ninputs              # Plot inputs separately
        noutput_axes = noutputs if plot_outputs else 0
        ninput_axes = ninputs if plot_inputs else 0

    ncols = ntraces if not overlay_traces else 1
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
        with plt.rc_context(timeplot_rcParams):
            ax_array = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
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
    # plot the data.  The arrays output_map and input_map are used to map
    # the different signals that are plotted onto the axes created above.
    # This code is complicated because it has to handle lots of different
    # variations.
    #

    # Create the map from trace, signal to axes, accounting for overlay_*
    output_map = np.empty((noutputs, ntraces), dtype=tuple)
    input_map = np.empty((ninputs, ntraces), dtype=tuple)

    for i in range(noutputs):
        for j in range(ntraces):
            signal_index = i if not overlay_signals else 0
            trace_index = j if not overlay_traces else 0
            if transpose:
                output_map[i, j] = (trace_index, signal_index + ninput_axes)
            else:
                output_map[i, j] = (signal_index, trace_index)

    for i in range(ninputs):
        for j in range(ntraces):
            signal_index = noutput_axes + (i if not overlay_signals else 0)
            trace_index = j if not overlay_traces else 0
            if transpose:
                input_map[i, j] = (trace_index, signal_index - noutput_axes)
            else:
                input_map[i, j] = (signal_index, trace_index)

    #
    # Plot the data
    #
    # The ax_output and ax_input arrays have the axes needed for making the
    # plots.  Labels are used on each axes for later creation of legends.
    # The gneric labels if of the form:
    #
    #     signal name, trace label, system name
    #
    # The signal name or tracel label can be omitted if they will appear on
    # the axes title or ylabel.  The system name is always included, since
    # multiple calls to plot() will require a legend that distinguishes
    # which system signals are plotted.  The system name is stripped off
    # later (in the legend-handling code) if it is not needed, but must be
    # included here since a plot may be built up by multiple calls to plot().
    #

    # Reshape the inputs and outputs for uniform indexing
    outputs = data.y.reshape(data.noutputs, ntraces, -1)
    if data.u is None or not plot_inputs:
        inputs = None
    else:
        inputs = data.u.reshape(data.ninputs, ntraces, -1)

    # Create a list of lines for the output
    out = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            out[i, j] = []      # unique list in each element

    # Utility function for creating line label
    def _make_line_label(signal_index, signal_labels, trace_index):
        label = ""              # start with an empty label

        # Add the signal name if it won't appear as an axes label
        if overlay_signals or plot_inputs == 'overlay':
            label += signal_labels[signal_index]

        # Add the trace label if this is a multi-trace figure
        if overlay_traces and ntraces > 1 or trace_labels:
            label += ", " if label != "" else ""
            if trace_labels:
                label += trace_labels[trace_index]
            elif data.trace_labels:
                label += data.trace_labels[trace_index]
            else:
                label += f"trace {trace_index}"

        # Add the system name (will strip off later if redundant)
        label += ", " if label != "" else ""
        label += f"{data.sysname}"

        return label

    # Go through each trace and each input/output
    for trace in range(ntraces):
        # Plot the output
        for i in range(noutputs):
            label = _make_line_label(i, data.output_labels, trace)

            # Set up line properties for this output, trace
            if len(fmt) == 0:
                line_props = output_props[
                    i % oprop_len if overlay_signals else 0].copy()
                line_props.update(
                    trace_props[trace % tprop_len if overlay_traces else 0])
                line_props.update(kwargs)
            else:
                line_props = kwargs

            out[output_map[i, trace]] += ax_array[output_map[i, trace]].plot(
                data.time, outputs[i][trace], *fmt, label=label, **line_props)

        # Plot the input
        for i in range(ninputs):
            label = _make_line_label(i, data.input_labels, trace)

            if add_initial_zero and data.trace_types \
               and data.trace_types[i] == 'step':
                x = np.hstack([np.array([data.time[0]]), data.time])
                y = np.hstack([np.array([0]), inputs[i][trace]])
            else:
                x, y = data.time, inputs[i][trace]

            # Set up line properties for this output, trace
            if len(fmt) == 0:
                line_props = input_props[
                    i % iprop_len if overlay_signals else 0].copy()
                line_props.update(
                    trace_props[trace % tprop_len if overlay_traces else 0])
                line_props.update(kwargs)
            else:
                line_props = kwargs

            out[input_map[i, trace]] += ax_array[input_map[i, trace]].plot(
                x, y, *fmt, label=label, **line_props)

    # Stop here if the user wants to control everything
    if not relabel:
        return out

    #
    # Label the axes (including trace labels)
    #
    # Once the data are plotted, we label the axes.  The horizontal axes is
    # always time and this is labeled only on the bottom most column.  The
    # vertical axes can consist either of a single signal or a combination
    # of signals (when overlay_signal is True or plot+inputs = 'overlay'.
    #
    # Traces are labeled at the top of the first row of plots (regular) or
    # the left edge of rows (tranpose).
    #

    # Time units on the bottom
    for col in range(ncols):
        ax_array[-1, col].set_xlabel(time_label)

    # Keep track of whether inputs are overlaid on outputs
    overlaid = plot_inputs == 'overlay'
    overlaid_title = "Inputs, Outputs"

    if transpose:               # inputs on left, outputs on right
        # Label the inputs
        if overlay_signals and plot_inputs:
            label = overlaid_title if overlaid else "Inputs"
            for trace in range(ntraces):
                ax_array[input_map[0, trace]].set_ylabel(label)
        else:
            for i in range(ninputs):
                label = overlaid_title if overlaid else data.input_labels[i]
                for trace in range(ntraces):
                    ax_array[input_map[i, trace]].set_ylabel(label)

        # Label the outputs
        if overlay_signals and plot_outputs:
            label = overlaid_title if overlaid else "Outputs"
            for trace in range(ntraces):
                ax_array[output_map[0, trace]].set_ylabel(label)
        else:
            for i in range(noutputs):
                label = overlaid_title if overlaid else data.output_labels[i]
                for trace in range(ntraces):
                    ax_array[output_map[i, trace]].set_ylabel(label)

        # Set the trace titles, if needed
        if ntraces > 1 and not overlay_traces:
            for trace in range(ntraces):
                # Get the existing ylabel for left column
                label = ax_array[trace, 0].get_ylabel()

                # Add on the trace title
                if trace_labels:
                    label = trace_labels[trace] + "\n" + label
                elif data.trace_labels:
                    label = data.trace_labels[trace] + "\n" + label
                else:
                    label = f"Trace {trace}" + "\n" + label

                ax_array[trace, 0].set_ylabel(label)

    else:                       # regular plot (outputs over inputs)
        # Set the trace titles, if needed
        if ntraces > 1 and not overlay_traces:
            for trace in range(ntraces):
                if trace_labels:
                    label = trace_labels[trace]
                elif data.trace_labels:
                    label = data.trace_labels[trace]
                else:
                    label = f"Trace {trace}"

                with plt.rc_context(timeplot_rcParams):
                    ax_array[0, trace].set_title(label)

        # Label the outputs
        if overlay_signals and plot_outputs:
            ax_array[output_map[0, 0]].set_ylabel("Outputs")
        else:
            for i in range(noutputs):
                ax_array[output_map[i, 0]].set_ylabel(
                    overlaid_title if overlaid else data.output_labels[i])

        # Label the inputs
        if overlay_signals and plot_inputs:
            label = overlaid_title if overlaid else "Inputs"
            ax_array[input_map[0, 0]].set_ylabel(label)
        else:
            for i in range(ninputs):
                label = overlaid_title if overlaid else data.input_labels[i]
                ax_array[input_map[i, 0]].set_ylabel(label)

    #
    # Create legends
    #
    # Legends can be placed manually by passing a legend_map array that
    # matches the shape of the suplots, with each item being a string
    # indicating the location of the legend for that axes (or None for no
    # legend).
    #
    # If no legend spec is passed, a minimal number of legends are used so
    # that each line in each axis can be uniquely identified.  The details
    # depends on the various plotting parameters, but the general rule is
    # to place legends in the top row and right column.
    #
    # Because plots can be built up by multiple calls to plot(), the legend
    # strings are created from the line labels manually.  Thus an initial
    # call to plot() may not generate any legends (eg, if no signals are
    # combined nor overlaid), but subsequent calls to plot() will need a
    # legend for each different line (system).
    #

    # Figure out where to put legends
    if legend_map is None:
        legend_map = np.full(ax_array.shape, None, dtype=object)
        if legend_loc == None:
            legend_loc = 'center right'
        if transpose:
            if (overlay_signals or plot_inputs == 'overlay') and overlay_traces:
                # Put a legend in each plot for inputs and outputs
                if plot_outputs is True:
                    legend_map[0, ninput_axes] = legend_loc
                if plot_inputs is True:
                    legend_map[0, 0] = legend_loc
            elif overlay_signals:
                # Put a legend in rightmost input/output plot
                if plot_inputs is True:
                    legend_map[0, 0] = legend_loc
                if plot_outputs is True:
                    legend_map[0, ninput_axes] = legend_loc
            elif plot_inputs == 'overlay':
                # Put a legend on the top of each column
                for i in range(ntraces):
                    legend_map[0, i] = legend_loc
            elif overlay_traces:
                # Put a legend topmost input/output plot
                legend_map[0, -1] = legend_loc
            else:
                # Put legend in the upper right
                legend_map[0, -1] = legend_loc
        else:                   # regular layout
            if (overlay_signals or plot_inputs == 'overlay') and overlay_traces:
                # Put a legend in each plot for inputs and outputs
                if plot_outputs is True:
                    legend_map[0, -1] = legend_loc
                if plot_inputs is True:
                    legend_map[noutput_axes, -1] = legend_loc
            elif overlay_signals:
                # Put a legend in rightmost input/output plot
                if plot_outputs is True:
                    legend_map[0, -1] = legend_loc
                if plot_inputs is True:
                    legend_map[noutput_axes, -1] = legend_loc
            elif plot_inputs == 'overlay':
                # Put a legend on the right of each row
                for i in range(max(ninputs, noutputs)):
                    legend_map[i, -1] = legend_loc
            elif overlay_traces:
                # Put a legend topmost input/output plot
                legend_map[0, -1] = legend_loc
            else:
                # Put legend in the upper right
                legend_map[0, -1] = legend_loc

    # Create axis legends
    for i in range(nrows):
        for j in range(ncols):
            ax = ax_array[i, j]
            # Get the labels to use
            labels = [line.get_label() for line in ax.get_lines()]

            # Look for a common prefix (up to a space)
            # TODO: fix error in 1x2, overlay, transpose (Fig 24)
            common_prefix = commonprefix(labels)
            last_space = common_prefix.rfind(', ')
            if last_space < 0 or plot_inputs == 'overlay':
                common_prefix = ''
            elif last_space > 0:
                common_prefix = common_prefix[:last_space]
            prefix_len = len(common_prefix)

            # Look for a common suffice (up to a space)
            common_suffix = commonprefix(
                [label[::-1] for label in labels])[::-1]
            suffix_len = len(common_suffix)
            # Only chop things off after a comma or space
            while suffix_len > 0 and common_suffix[-suffix_len] != ',':
                suffix_len -= 1

            # Strip the labels of common information
            if suffix_len > 0:
                labels = [label[prefix_len:-suffix_len] for label in labels]
            else:
                labels = [label[prefix_len:] for label in labels]

            # Update the labels to remove common strings
            if len(labels) > 1 and legend_map[i, j] != None:
                with plt.rc_context(timeplot_rcParams):
                    ax.legend(labels, loc=legend_map[i, j])

    #
    # Update the plot title (= figure suptitle)
    #
    # If plots are built up by multiple calls to plot() and the title is
    # not given, then the title is updated to provide a list of unique text
    # items in each successive title.  For data generated by the I/O
    # response functions this will generate a common prefix followed by a
    # list of systems (e.g., "Step response for sys[1], sys[2]").
    #

    if fig is not None and title is not None:
        # Get the current title, if it exists
        old_title = None if fig._suptitle is None else fig._suptitle._text
        new_title = title

        if old_title is not None:
            # Find the common part of the titles
            common_prefix = commonprefix([old_title, new_title])

            # Back up to the last space
            last_space = common_prefix.rfind(' ')
            if last_space > 0:
                common_prefix = common_prefix[:last_space]
            common_len = len(common_prefix)

            # Add the new part of the title (usually the system name)
            if old_title[common_len:] != new_title[common_len:]:
                separator = ',' if len(common_prefix) > 0 else ';'
                new_title = old_title + separator + new_title[common_len:]

        # Add the title
        with plt.rc_context(timeplot_rcParams):
            fig.suptitle(new_title)

    return out


def combine_traces(response_list, trace_labels=None, title=None):
    """Combine multiple individual time responses into a multi-trace response.

    This function combines multiple instances of :class:`TimeResponseData`
    into a multi-trace :class:`TimeResponseData` object.

    Parameters
    ----------
    response_list : list of :class:`TimeResponseData` objects
        Reponses to be combined.
    trace_labels : list of str, optional
        List of labels for each trace.  If not specified, trace names are
        taken from the input data or set to None.

    Returns
    -------
    data : :class:`TimeResponseData`
        Multi-trace input/output data.

    """
    from .timeresp import TimeResponseData

    # Save the first trace as the base case
    base = response_list[0]

    # Process keywords
    title = base.title if title is None else title

    # Figure out the size of the data (and check for consistency)
    ntraces = max(1, base.ntraces)

    # Initial pass through trace list to count things up and do error checks
    for response in response_list[1:]:
        # Make sure the time vector is the same
        if not np.allclose(base.t, response.t):
            raise ValueError("all responses must have the same time vector")

        # Make sure the dimensions are all the same
        if base.ninputs != response.ninputs or \
           base.noutputs != response.noutputs or \
           base.nstates != response.nstates:
            raise ValueError("all responses must have the same number of "
                            "inputs, outputs, and states")

        ntraces += max(1, response.ntraces)

    # Create data structures for the new time response data object
    inputs = np.empty((base.ninputs, ntraces, base.t.size))
    outputs = np.empty((base.noutputs, ntraces, base.t.size))
    states = np.empty((base.nstates, ntraces, base.t.size))

    # See whether we should create labels or not
    if trace_labels is None:
        generate_trace_labels = True
        trace_labels = []
    elif len(trace_labels) != ntraces:
        raise ValueError(
            "number of trace labels does not match number of traces")
    else:
        generate_trace_labels = False

    offset = 0
    trace_types = []
    for response in response_list:
        if response.ntraces == 0:
            # Single trace
            inputs[:, offset, :] = response.u
            outputs[:, offset, :] = response.y
            states[:, offset, :] = response.x
            offset += 1

            # Add on trace label and trace type
            if generate_trace_labels:
                trace_labels.append(response.title)
            trace_types.append(
                None if response.trace_types is None else response.types[0])

        else:
            # Save the data
            for i in range(response.ntraces):
                inputs[:, offset, :] = response.u[:, i, :]
                outputs[:, offset, :] = response.y[:, i, :]
                states[:, offset, :] = response.x[:, i, :]

                # Save the trace labels
                if generate_trace_labels:
                    if response.trace_labels is not None:
                        trace_labels.append(response.trace_labels[i])
                    else:
                        trace_labels.append(response.title + f", trace {i}")

                offset += 1

            # Save the trace types
            if response.trace_types is not None:
                trace_types += response.trace_types
            else:
                trace_types += [None] * response.ntraces

    return TimeResponseData(
        base.t, outputs, states, inputs, issiso=base.issiso,
        output_labels=base.output_labels, input_labels=base.input_labels,
        state_labels=base.state_labels, title=title, transpose=base.transpose,
        return_x=base.return_x, squeeze=base.squeeze, sysname=base.sysname,
        trace_labels=trace_labels, trace_types=trace_types,
        plot_inputs=base.plot_inputs)


# Create vectorized function to find axes from lines
def get_plot_axes(line_array):
    """Get a list of axes from an array of lines.

    This function can be used to return the set of axes corresponding to
    the line array that is returned by `time_response_plot`.  This is useful for
    generating an axes array that can be passed to subsequent plotting
    calls.

    Parameters
    ----------
    line_array : array of list of Line2D
        A 2D array with elements corresponding to a list of lines appearing
        in an axes, matching the return type of a time response data plot.

    Returns
    -------
    axes_array : array of list of Axes
        A 2D array with elements corresponding to the Axes assocated with
        the lines in `line_array`.

    Notes
    -----
    Only the first element of each array entry is used to determine the axes.

    """
    _get_axes = np.vectorize(lambda lines: lines[0].axes)
    return _get_axes(line_array)
