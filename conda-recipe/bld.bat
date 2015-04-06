cd %RECIPE_DIR%\..
%PYTHON% make_version.py
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
