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

"""Data type handling for state vectors."""

from __future__ import annotations

import numpy as np


def as_double_precision(vec: np.ndarray) -> np.ndarray:
    """Cast a vector to double precision, keeping it real or complex.

    Used before calling PySCF FCI kernels, which only handle float64 data.
    """
    vec = np.asarray(vec)
    target = np.complex128 if np.iscomplexobj(vec) else np.float64
    return vec.astype(target, copy=False)
