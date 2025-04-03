"""
This file is using ControlSystems.jl library to produce results and plots about delay systems. 
Theses results are then exported to json format, and compared to python-control pure delays implementation, 
as a way to benchmark it. The script and the json follows the test progression of the delay_lti_test.py file. 

In order to run this notebook, the user should install julia and the ControlSystems.jl library:
-  https://julialang.org/downloads/ 
-  https://github.com/JuliaControl/ControlSystems.jl
-  https://github.com/JuliaIO/JSON.jl
-  https://github.com/JuliaPlots/Plots.jl
"""

using ControlSystems
using Plots
using JSON

# ------------------
# Transfer functions
# ------------------

s = tf("s")

simple_siso_tf = tf([1], [1, 1])

tf_one = tf([1], [1])

delay_siso_tf = 1 / (s + 1) * delay(1.5)

delay_siso_tf2 = 3 / (2 * s + 5) * delay(0.5)

delay_tito_tf = [1/(s+1)*exp(-0.5*s) 1/(s+2)*exp(-s);  1/(s+3)*exp(-s)  1/(s+4)*exp(-s)]

wood_berry = [12.8/(16.7s+1)*exp(-s) -18.9/(21s+1)*exp(-3s);  6.6/(10.9s+1)*exp(-7s)  -19.4/(14.4s+1)*exp(-3s)]


function dlti2dict(dlti)
    return Dict(
        "A" => Dict(
            "data" => dlti.P.A,
            "dim" => size(dlti.P.A)
        ),
        "B" => Dict(
            "data" => dlti.P.B,
            "dim" => size(dlti.P.B)
        ),
        "C" => Dict(
            "data" => dlti.P.C,
            "dim" => size(dlti.P.C)
        ),
        "D" => Dict(
            "data" => dlti.P.D,
            "dim" => size(dlti.P.D)
        ),
        "tau" => Dict(
            "data" => dlti.Tau,
            "dim" => size(dlti.Tau)
        )
    )
end


function test_tf2dlti(tf)
    dlti = DelayLtiSystem(tf)
    return dlti2dict(dlti)
end


function test_delay_function(tau)
    dlti = delay(tau)
    return dlti2dict(dlti)
end


function test_exp_delay(tau)
    dlti = exp(-tau * s)
    return dlti2dict(dlti)
end

function complex_array_to_dict(arr)
    return Dict(
        "real" => real(arr),
        "imag" => imag(arr)
    )
end

function test_siso_freq_resp(tf)
    w = exp10.(LinRange(-2,2,100))
    arr = collect(Iterators.Flatten(freqresp(tf, w)))
    return complex_array_to_dict(arr)
end

function test_tito_freq_response(tf)
    w = exp10.(LinRange(-2,2,100))
    resp = freqresp(tf, w)
    resp_11 = resp[1, 1, :]
    resp_12 = resp[1, 2, :]
    resp_21 = resp[2, 1, :]
    resp_22 = resp[2, 2, :]

    return Dict(
        "r11" => complex_array_to_dict(resp_11),
        "r12" => complex_array_to_dict(resp_12),
        "r21" => complex_array_to_dict(resp_21),
        "r22" => complex_array_to_dict(resp_22),
    )
end

function main()

    results_TestConstructors = Dict(
        "test_tf2dlti" => Dict(
            "simple_siso_tf" => test_tf2dlti(simple_siso_tf),
            "tf_one" => test_tf2dlti(tf_one)
        ),
        "test_delay_function" => Dict(
            "1" => test_delay_function(1),
            "1.5" => test_delay_function(1.5),
            "10" => test_delay_function(10)
        ),
        "test_exp_delay" => Dict(
            "1" => test_exp_delay(1),
            "1.5" => test_exp_delay(1.5),
            "10" => test_exp_delay(10)
        ),
        "test_siso_delay" => dlti2dict(delay_siso_tf),
        "test_build_wood_berry" => dlti2dict(wood_berry)
    )

    results_TestOperators = Dict(
        "test_siso_add" => dlti2dict(delay_siso_tf + delay_siso_tf2),
        "test_siso_add_constant" => dlti2dict(delay_siso_tf + 2.5),
        "test_siso_sub" => dlti2dict(delay_siso_tf - delay_siso_tf2),
        "test_siso_sub_constant" => dlti2dict(delay_siso_tf - 2.5),
        "test_siso_mul" => dlti2dict(delay_siso_tf * delay_siso_tf2),
        "test_siso_mul_constant" => dlti2dict(delay_siso_tf * 2.),
        "test_siso_rmul_constant" => dlti2dict(2. * delay_siso_tf),
        "test_mimo_add" => dlti2dict(wood_berry + wood_berry),
        "test_mimo_add_constant" => dlti2dict(wood_berry + 2.7),
        "test_mimo_mul" => dlti2dict(wood_berry * wood_berry),
        "test_mimo_mul_constant" => dlti2dict(wood_berry * 2.7)
    )

    results_TestDelayLtiMethods = Dict(
        "test_feedback" => Dict(
            "empty" => dlti2dict(feedback(delay_siso_tf, 1)),
            "tf_one" => dlti2dict(feedback(delay_siso_tf, tf_one)),
            "delay_siso_tf" => dlti2dict(feedback(delay_siso_tf, delay_siso_tf))
        ),
        "test_mimo_feedback" => dlti2dict(feedback(wood_berry, wood_berry)),

        "test_siso_freq_resp" => test_siso_freq_resp(delay_siso_tf),
        "test_tito_freq_response" => test_tito_freq_response(wood_berry),

    )

    results = Dict(
        "TestConstructors" => results_TestConstructors,
        "TestOperators" => results_TestOperators,
        "TestDelayLtiMethods" => results_TestDelayLtiMethods,
    )

    script_dir = @__DIR__
    output_file = joinpath(script_dir, "julia_results.json")
    open(output_file, "w") do io
        JSON.print(io, results, 4)
    end

    println("Expected results exported to julia_results.json")
end

main()