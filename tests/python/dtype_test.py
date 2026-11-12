# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for state vector data type handling."""

from __future__ import annotations

import numpy as np

import ffsim

RNG = np.random.default_rng(223104777685777786317612746867353875493)


def test_linear_operator_single_precision():
    """Linear operators compute in double precision for single-precision input."""
    norb, nelec = 5, (3, 2)
    dim = ffsim.dim(norb, nelec)
    linops = [
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian(norb, seed=RNG), norb, nelec
        ),
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian_unrestricted(
                norb, seed=RNG, dtype=float
            ),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_double_factorized_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_single_factorized_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.fermion_operator(
                ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
            ),
            norb,
            nelec,
        ),
        ffsim.contract.one_body_linop(
            ffsim.random.random_hermitian(norb, seed=RNG), norb, nelec
        ),
    ]
    for linop in linops:
        for dtype, double_dtype in [
            (np.complex64, np.complex128),
            (np.float32, np.float64),
        ]:
            vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=dtype)
            result = linop @ vec
            expected = linop @ vec.astype(double_dtype)
            assert result.dtype == expected.dtype
            np.testing.assert_allclose(result, expected)
