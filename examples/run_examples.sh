#!/bin/bash
# Run the examples.  Doesn't run Jupyter notebooks.

# The examples must cooperate: they must have no operations that wait
# on user action, like matplotlib.pyplot.show, or starting a GUI event
# loop.  Environment variable PYCONTROL_TEST_EXAMPLES is set when the
# examples are being tested; existence of this variable should be used
# to prevent such user-action-waiting operations.

export PYCONTROL_TEST_EXAMPLES=1

example_errors=""

for example in *.py; do
    echo "Running ${example}"
    if ! python ${example}; then
        example_errors="${example_errors} ${example}"
    fi
done
    
# Get rid of the output files
rm *.log

# List any files that generated errors
if [ -n "${example_errors}" ]; then
    echo These examples had errors:
    echo "${example_errors}"
    exit 1
else
    echo All examples ran successfully
fi
