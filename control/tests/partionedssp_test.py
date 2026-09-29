import numpy as np
import pytest
from control.statesp import ss
from control.partitionedssp import PartitionedStateSpace, vcat_pss, hcat_pss


class TestPartitionedStateSpace:
    def test_init(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)

        assert np.array_equal(pss.A, A)
        assert np.array_equal(pss.B, B)
        assert np.array_equal(pss.C, C)
        assert np.array_equal(pss.D, D)
        assert np.array_equal(pss.B1, B[:, :1])
        assert np.array_equal(pss.B2, B[:, 1:])
        assert np.array_equal(pss.C1, C[:1, :])
        assert np.array_equal(pss.C2, C[1:, :])
        assert np.array_equal(pss.D11, D[:1, :1])
        assert np.array_equal(pss.D12, D[:1, 1:])
        assert np.array_equal(pss.D21, D[1:, :1])
        assert np.array_equal(pss.D22, D[1:, 1:])
        assert pss.nu1 == 1
        assert pss.ny1 == 1
        assert pss.nu2 == 1
        assert pss.ny2 == 1
        assert pss.nstates == 2
        assert pss.ninputs_total == 2
        assert pss.noutputs_total == 2

    def test_init_invalid_input(self):
        with pytest.raises(TypeError):
            PartitionedStateSpace("not a StateSpace", 1, 1)

    def test_init_invalid_nu1(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        with pytest.raises(ValueError):
            PartitionedStateSpace(sys, 3, 1)
        with pytest.raises(ValueError):
            PartitionedStateSpace(sys, -1, 1)

    def test_init_invalid_ny1(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        with pytest.raises(ValueError):
            PartitionedStateSpace(sys, 1, 3)
        with pytest.raises(ValueError):
            PartitionedStateSpace(sys, 1, -1)

    def test_from_matrices(self):
        A = np.array([[1, 2], [3, 4]])
        B1 = np.array([[5], [7]])
        B2 = np.array([[6], [8]])
        C1 = np.array([[9, 10]])
        C2 = np.array([[11, 12]])
        D11 = np.array([[13]])
        D12 = np.array([[14]])
        D21 = np.array([[15]])
        D22 = np.array([[16]])

        pss = PartitionedStateSpace.from_matrices(A, B1, B2, C1, C2, D11, D12, D21, D22)

        assert np.array_equal(pss.A, A)
        assert np.array_equal(pss.B1, B1)
        assert np.array_equal(pss.B2, B2)
        assert np.array_equal(pss.C1, C1)
        assert np.array_equal(pss.C2, C2)
        assert np.array_equal(pss.D11, D11)
        assert np.array_equal(pss.D12, D12)
        assert np.array_equal(pss.D21, D21)
        assert np.array_equal(pss.D22, D22)
        assert pss.nu1 == 1
        assert pss.ny1 == 1
        assert pss.nu2 == 1
        assert pss.ny2 == 1
        assert pss.nstates == 2
        assert pss.ninputs_total == 2
        assert pss.noutputs_total == 2

    def test_from_matrices_invalid_shapes(self):
        A = np.array([[1, 2], [3, 4]])
        B1 = np.array([[5], [7]])
        B2 = np.array([[6], [8]])
        C1 = np.array([[9, 10]])
        C2 = np.array([[11, 12]])
        D11 = np.array([[13]])
        D12 = np.array([[14]])
        D21 = np.array([[15]])
        D22 = np.array([[16]])

        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, C1, C2, D11, D12, D21, np.array([[16, 17]])
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, C1, C2, D11, D12, np.array([[15, 16]]), D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, C1, C2, D11, np.array([[14, 15]]), D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, C1, C2, np.array([[13, 14]]), D12, D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, C1, np.array([[11, 12, 13]]), D11, D12, D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, B2, np.array([[9, 10, 11]]), C2, D11, D12, D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, B1, np.array([[6, 7], [8, 9]]), C1, C2, D11, D12, D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                A, np.array([[5, 6], [7, 8]]), B2, C1, C2, D11, D12, D21, D22
            )
        with pytest.raises(ValueError):
            PartitionedStateSpace.from_matrices(
                np.array([[1, 2, 3], [4, 5, 6]]), B1, B2, C1, C2, D11, D12, D21, D22
            )

    def test_add_invalid_type(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)
        with pytest.raises(TypeError):
            pss + "not a PartitionedStateSpace"

    def test_mul_invalid_type(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)
        with pytest.raises(TypeError):
            pss * "not a PartitionedStateSpace"

    def test_feedback_invalid_type(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)
        with pytest.raises(TypeError):
            pss.feedback("not a PartitionedStateSpace")

    def test_vcat_pss(self):
        A1 = np.array([[1, 2], [3, 4]])
        B1_1 = np.array([[5], [7]])
        B1_2 = np.array([[6], [8]])
        C1_1 = np.array([[9, 10]])
        C1_2 = np.array([[11, 12]])
        D11_1 = np.array([[13]])
        D12_1 = np.array([[14]])
        D21_1 = np.array([[15]])
        D22_1 = np.array([[16]])
        pss1 = PartitionedStateSpace.from_matrices(
            A1, B1_1, B1_2, C1_1, C1_2, D11_1, D12_1, D21_1, D22_1
        )

        A2 = np.array([[1, 2], [3, 4]])
        B2_1 = np.array([[5], [7]])
        B2_2 = np.array([[6], [8]])
        C2_1 = np.array([[9, 10]])
        C2_2 = np.array([[11, 12]])
        D11_2 = np.array([[13]])
        D12_2 = np.array([[14]])
        D21_2 = np.array([[15]])
        D22_2 = np.array([[16]])
        pss2 = PartitionedStateSpace.from_matrices(
            A2, B2_1, B2_2, C2_1, C2_2, D11_2, D12_2, D21_2, D22_2
        )

        pss_vcat = vcat_pss(pss1, pss2)

        assert np.array_equal(
            pss_vcat.A, np.block([[A1, np.zeros_like(A1)], [np.zeros_like(A2), A2]])
        )
        assert np.array_equal(pss_vcat.B1, np.vstack((B1_1, B2_1)))
        assert np.array_equal(
            pss_vcat.B2,
            np.block([[B1_2, np.zeros_like(B2_2)], [np.zeros_like(B1_2), B2_2]]),
        )
        assert np.array_equal(
            pss_vcat.C1,
            np.block([[C1_1, np.zeros_like(C2_1)], [np.zeros_like(C1_1), C2_1]]),
        )
        assert np.array_equal(
            pss_vcat.C2,
            np.block([[C1_2, np.zeros_like(C2_2)], [np.zeros_like(C1_2), C2_2]]),
        )
        assert np.array_equal(pss_vcat.D11, np.vstack((D11_1, D11_2)))
        assert np.array_equal(
            pss_vcat.D12,
            np.block([[D12_1, np.zeros_like(D12_2)], [np.zeros_like(D12_1), D12_2]]),
        )
        assert np.array_equal(pss_vcat.D21, np.vstack((D21_1, D21_2)))
        assert np.array_equal(
            pss_vcat.D22,
            np.block([[D22_1, np.zeros_like(D22_2)], [np.zeros_like(D22_1), D22_2]]),
        )
        assert pss_vcat.nu1 == 1
        assert pss_vcat.ny1 == 2
        assert pss_vcat.nu2 == 2
        assert pss_vcat.ny2 == 2
        assert pss_vcat.nstates == 4
        assert pss_vcat.ninputs_total == 3
        assert pss_vcat.noutputs_total == 4

    def test_vcat_pss_invalid_type(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)
        with pytest.raises(TypeError):
            vcat_pss(pss, "not a PartitionedStateSpace")

    def test_vcat_pss_invalid_input_dimension(self):
        A1 = np.array([[1, 2], [3, 4]])
        B1_1 = np.array([[5], [7]])
        B1_2 = np.array([[6], [8]])
        C1_1 = np.array([[9, 10]])
        C1_2 = np.array([[11, 12]])
        D11_1 = np.array([[13]])
        D12_1 = np.array([[14]])
        D21_1 = np.array([[15]])
        D22_1 = np.array([[16]])
        pss1 = PartitionedStateSpace.from_matrices(
            A1, B1_1, B1_2, C1_1, C1_2, D11_1, D12_1, D21_1, D22_1
        )

        A2 = np.array([[1, 2], [3, 4]])
        B2_1 = np.array([[5, 6], [7, 8]])
        B2_2 = np.array([[6, 7], [8, 9]])
        C2_1 = np.array([[9, 10]])
        C2_2 = np.array([[11, 12]])
        D11_2 = np.array([[13, 14]])
        D12_2 = np.array([[14, 15]])
        D21_2 = np.array([[15, 16]])
        D22_2 = np.array([[16, 17]])
        pss2 = PartitionedStateSpace.from_matrices(
            A2, B2_1, B2_2, C2_1, C2_2, D11_2, D12_2, D21_2, D22_2
        )

        with pytest.raises(ValueError):
            vcat_pss(pss1, pss2)

    def test_hcat_pss(self):
        A1 = np.array([[1, 2], [3, 4]])
        B1_1 = np.array([[5], [7]])
        B1_2 = np.array([[6], [8]])
        C1_1 = np.array([[9, 10]])
        C1_2 = np.array([[11, 12]])
        D11_1 = np.array([[13]])
        D12_1 = np.array([[14]])
        D21_1 = np.array([[15]])
        D22_1 = np.array([[16]])
        pss1 = PartitionedStateSpace.from_matrices(
            A1, B1_1, B1_2, C1_1, C1_2, D11_1, D12_1, D21_1, D22_1
        )

        A2 = np.array([[1, 2], [3, 4]])
        B2_1 = np.array([[5], [7]])
        B2_2 = np.array([[6], [8]])
        C2_1 = np.array([[9, 10]])
        C2_2 = np.array([[11, 12]])
        D11_2 = np.array([[13]])
        D12_2 = np.array([[14]])
        D21_2 = np.array([[15]])
        D22_2 = np.array([[16]])
        pss2 = PartitionedStateSpace.from_matrices(
            A2, B2_1, B2_2, C2_1, C2_2, D11_2, D12_2, D21_2, D22_2
        )

        pss_hcat = hcat_pss(pss1, pss2)

        assert np.array_equal(
            pss_hcat.A, np.block([[A1, np.zeros_like(A1)], [np.zeros_like(A2), A2]])
        )
        assert np.array_equal(
            pss_hcat.B1,
            np.block([[B1_1, np.zeros_like(B2_1)], [np.zeros_like(B1_1), B2_1]]),
        )
        assert np.array_equal(
            pss_hcat.B2,
            np.block([[B1_2, np.zeros_like(B2_2)], [np.zeros_like(B1_2), B2_2]]),
        )
        assert np.array_equal(pss_hcat.C1, np.hstack((C1_1, C2_1)))
        assert np.array_equal(
            pss_hcat.C2,
            np.block([[C1_2, np.zeros_like(C2_2)], [np.zeros_like(C1_2), C2_2]]),
        )
        assert np.array_equal(pss_hcat.D11, np.hstack((D11_1, D11_2)))
        assert np.array_equal(pss_hcat.D12, np.hstack((D12_1, D12_2)))
        assert np.array_equal(
            pss_hcat.D21,
            np.block([[D21_1, np.zeros_like(D21_2)], [np.zeros_like(D21_1), D21_2]]),
        )
        assert np.array_equal(
            pss_hcat.D22,
            np.block([[D22_1, np.zeros_like(D22_2)], [np.zeros_like(D22_1), D22_2]]),
        )
        assert pss_hcat.nu1 == 2
        assert pss_hcat.ny1 == 1
        assert pss_hcat.nu2 == 2
        assert pss_hcat.ny2 == 2
        assert pss_hcat.nstates == 4
        assert pss_hcat.ninputs_total == 4
        assert pss_hcat.noutputs_total == 3

    def test_hcat_pss_invalid_type(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.array([[9, 10], [11, 12]])
        D = np.array([[13, 14], [15, 16]])
        sys = ss(A, B, C, D)
        pss = PartitionedStateSpace(sys, 1, 1)
        with pytest.raises(TypeError):
            hcat_pss(pss, "not a PartitionedStateSpace")

    def test_hcat_pss_invalid_output_dimension(self):
        A1 = np.array([[1, 2], [3, 4]])
        B1_1 = np.array([[5], [7]])
        B1_2 = np.array([[6], [8]])
        C1_1 = np.array([[9, 10]])
        C1_2 = np.array([[11, 12]])
        D11_1 = np.array([[13]])
        D12_1 = np.array([[14]])
        D21_1 = np.array([[15]])
        D22_1 = np.array([[16]])
        pss1 = PartitionedStateSpace.from_matrices(
            A1, B1_1, B1_2, C1_1, C1_2, D11_1, D12_1, D21_1, D22_1
        )

        A2 = np.array([[1, 2], [3, 4]])
        B2_1 = np.array([[5], [7]])
        B2_2 = np.array([[6], [8]])
        C2_1 = np.array([[9, 10], [11, 12]])
        C2_2 = np.array([[11, 12], [13, 14]])
        D11_2 = np.array([[13], [14]])
        D12_2 = np.array([[14], [15]])
        D21_2 = np.array([[15], [16]])
        D22_2 = np.array([[16], [17]])
        pss2 = PartitionedStateSpace.from_matrices(
            A2, B2_1, B2_2, C2_1, C2_2, D11_2, D12_2, D21_2, D22_2
        )

        with pytest.raises(ValueError):
            hcat_pss(pss1, pss2)
