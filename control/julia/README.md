# Use of Julia for time delay implementation

The implementation of continuous time delays was done by porting some functionalities of ControlSystems.jl to python. So it seemed natural to compare results from python to this library.

The ``compute_tests.jl`` file follows the structure of the ``delay_lti_test.py`` file to produce results and plots about delay systems. Theses results are then exported to json format, and compared to python-control pure delays implementation, as a way to benchmark it. 

In order to run the ``compute_tests.jl`` file, the user should install:
-  the julia REPL from https://julialang.org/downloads/ 
-  the ControlSystems.jl package from https://github.com/JuliaControl/ControlSystems.jl
-  the JSON.jl package from https://github.com/JuliaIO/JSON.jl

Then, the user should open a terminal:
```bash
cd <path_to_control_package>/control/julia
julia compute_tests.jl 
```

The ``utils.py`` file contains helper functions to deserialize data from json to python.