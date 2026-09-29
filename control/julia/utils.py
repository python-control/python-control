import numpy as np
import json

from pathlib import Path


def _recursive_reshape(node):
    """Reshape arrays in a nested dictionary or list.

    This function recursively traverses a nested dictionary or list,
    looking for dictionaries with "data" and "dim" keys. If found, it
    reshapes the "data" array according to the "dim" tuple.

    Parameters
    ----------
    node : dict or list
        The nested dictionary or list to process.

    Returns
    -------
    dict or list
        The processed dictionary or list with reshaped arrays.
    """

    if isinstance(node, dict) and ("data" in node) and ("dim" in node):
        data = node["data"]
        dim = tuple(node["dim"])

        array_data = np.array(data)

        if len(dim) == 1:
            return array_data

        elif len(dim) > 1 and (np.shape(array_data) == dim) and (dim[0] != dim[1]):
            return array_data
        else:
            return np.transpose(array_data)

    elif isinstance(node, dict):
        new_node = {}
        for key, value in node.items():
            new_node[key] = _recursive_reshape(value)
        return new_node

    elif isinstance(node, list) and any(isinstance(item, dict) for item in node):
        new_node = []
        for i, item in enumerate(node):
            new_node[i] = _recursive_reshape(item)
        return new_node

    else:
        return node


def load_julia_results(json_path: str):
    """Load Julia results from a JSON file and reshape arrays.

    This function loads data from a JSON file, which is assumed to
    contain results from a Julia simulation. It then reshapes arrays
    within the loaded data using the _recursive_reshape function.

    Parameters
    ----------
    json_path : str
        The path to the JSON file containing the Julia results.

    Returns
    -------
    dict
        The reshaped Julia results.
    """

    with open(json_path, "r") as f:
        json_content = json.load(f)

    reshaped_julia = _recursive_reshape(json_content)
    return reshaped_julia


def assert_delayLTI(dlti, julia_results):
    """Assert that a DelayLTI object matches the Julia results.

    Parameters
    ----------
    dlti : DelayLTI
        The DelayLTI object to compare.
    julia_results : dict
        The Julia results to compare against.
    """
    assert np.allclose(dlti.P.A, julia_results["A"])
    assert np.allclose(dlti.P.B, julia_results["B"])
    assert np.allclose(dlti.P.C, julia_results["C"])
    assert np.allclose(dlti.P.D, julia_results["D"])
    assert np.allclose(dlti.tau, julia_results["tau"])


# Load julia results file
script_dir = Path(__file__).parent
julia_json = load_julia_results(f"{script_dir}/julia_results.json")
