#!/bin/bash
# run_notbooks.sh - run Jupyter notebooks
# RMM, 26 Jan 2021
#
# This script runs all of the Jupyter notebooks to make sure that there
# are no errors.

# Keep track of files that generate errors
notebook_errors=""

# Go through each Jupyter notebook
for example in *.ipynb; do
    echo "Running ${example}"
    if ! jupyter nbconvert --to notebook --execute ${example}; then
        notebook_errors="${notebook_errors} ${example}"
    fi
done

# Get rid of the output files
rm *.nbconvert.ipynb

# List any files that generated errors
if [ -n "${notebook_errors}" ]; then
    echo These examples had errors:
    echo "${notebook_errors}"
    exit 1
else
    echo All examples ran successfully
fi
