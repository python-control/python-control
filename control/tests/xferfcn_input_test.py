"""xferfcn_input_test.py - test inputs to TransferFunction class

jed-frey, 18 Feb 2017 (based on xferfcn_test.py)
BG,       31 Jul 2020 convert to pytest and parametrize into single function
"""

import numpy as np
import pytest

from control.xferfcn import _clean_part

cases = {
    "scalar":
        (1, lambda dtype, v: dtype(v)),
    "scalar in 0d array":
        (1, lambda dtype, v: np.array(v, dtype=dtype)),
    "numpy array":
        ([1, 2], lambda dtype, v: np.array(v, dtype=dtype)),
    "list of scalar":
        (1, lambda dtype, v: [dtype(v)]),
    "list of scalars":
        ([1, 2], lambda dtype, v: [dtype(vi) for vi in v]),
    "list of list of list of scalar":
        (1, lambda dtype, v: [[[dtype(v)]]]),
    "list of list of list of scalars":
        ([[1, 1], [2, 2]],
         lambda dtype, v: [[[dtype(vi) for vi in vr] for vr in v]]),
    "tuple of scalar":
        (1, lambda dtype, v: (dtype(v),)),
    "tuple of scalars":
        ([1, 2], lambda dtype, v: tuple(dtype(vi) for vi in v)),
    "list of list of numpy arrays":
        ([[1, 1], [2, 2]],
         lambda dtype, v: [[np.array(vr, dtype=dtype) for vr in v]]),
    "tuple of list of numpy arrays":
        ([[1, 1], [2, 2]],
         lambda dtype, v: ([np.array(vr, dtype=dtype) for vr in v],)),
    "list of tuple of numpy arrays":
        ([[1, 1], [2, 2]],
         lambda dtype, v: [tuple(np.array(vr, dtype=dtype) for vr in v)]),
    "tuple of tuples of numpy arrays":
        ([[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
         lambda dtype, v: tuple(tuple(np.array(vr, dtype=dtype) for vr in vp)
                                for vp in v)),
    "list of tuples of numpy arrays":
        ([[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
         lambda dtype, v: [tuple(np.array(vr, dtype=dtype) for vr in vp)
                           for vp in v]),
    "list of lists of numpy arrays":
        ([[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
         lambda dtype, v: [[np.array(vr, dtype=dtype) for vr in vp]
                           for vp in v]),
}


@pytest.mark.parametrize("dtype",
                         [int, np.int8, np.int16, np.int32, np.int64,
                          float, np.float16, np.float32, np.float64,
                          np.longdouble])
@pytest.mark.parametrize("num, fun", cases.values(), ids=cases.keys())
def test_clean_part(num, fun, dtype):
    """Test clean part for various inputs"""
    numa = fun(dtype, num)
    num_ = _clean_part(numa)
    ref_ = np.array(num, dtype=float, ndmin=3)

    assert isinstance(num_, list)
    assert np.all([isinstance(part, list) for part in num_])
    for i, numi in enumerate(num_):
        assert len(numi) == ref_.shape[1]
        for j, numj in enumerate(numi):
            np.testing.assert_allclose(numj, ref_[i, j, ...])


@pytest.mark.parametrize("badinput", [[[0., 1.], [2., 3.]], "a"])
def test_clean_part_bad_input(badinput):
    """Give the part cleaner invalid input type."""
    with pytest.raises(TypeError):
        _clean_part(badinput)
