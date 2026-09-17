# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for serialization."""

from __future__ import annotations

import dataclasses
import gzip
import json
import pathlib

import numpy as np
import pytest
from pyscf.fci.direct_spin1 import FCIvector

import ffsim
from ffsim.serialization import _REGISTRY, _register_serializable

RNG = np.random.default_rng(133484729805520748291743062596228043017)

COMPRESSIONS = [None, "gzip", "bz2", "lzma"]


@_register_serializable(name="test.SerializationValue")
@dataclasses.dataclass
class SerializationValue:
    value: object


def assert_exactly_equal(actual, expected) -> None:
    """Compare types and values recursively, including array dtypes and shapes."""
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, ffsim.FermionOperator):
        assert actual == expected
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            assert_exactly_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_exactly_equal(actual_item, expected_item)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key, expected_item in expected.items():
            assert_exactly_equal(actual[key], expected_item)
    else:
        assert actual == expected


def _molecular_data() -> ffsim.MolecularData:
    norb = 3
    return ffsim.MolecularData(
        core_energy=1.5,
        one_body_integrals=ffsim.random.random_hermitian(norb, seed=RNG),
        two_body_integrals=ffsim.random.random_two_body_tensor(norb, seed=RNG),
        norb=norb,
        nelec=(2, 1),
        atom=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.75])],
        basis="sto-6g",
        active_space=[0, 1, 2],
        hf_energy=-1.25,
        # An unrestricted amplitude field, which is a tuple of arrays rather than an
        # array.
        ccsd_t1=(
            ffsim.random.random_hermitian(norb, seed=RNG),
            ffsim.random.random_hermitian(norb, seed=RNG),
        ),
        orbital_symmetries=["A1", "A1", "B1"],
    )


def _objects() -> dict[str, object]:
    """Return one instance of every serializable type, keyed by a test id."""
    norb = 4
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (2, 2)), seed=RNG)
    one_rdm, two_rdm = ffsim.rdms(vec, norb, (2, 2), rank=2)
    mol_data = _molecular_data()
    return {
        # Hamiltonians
        "MolecularHamiltonian": mol_hamiltonian,
        "MolecularHamiltonianSpinless": (
            ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=RNG)
        ),
        "MolecularHamiltonianUnrestricted": (
            ffsim.random.random_molecular_hamiltonian_unrestricted(norb, seed=RNG)
        ),
        "DiagonalCoulombHamiltonian": (
            ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG)
        ),
        "DoubleFactorizedHamiltonian": (
            ffsim.random.random_double_factorized_hamiltonian(norb, seed=RNG)
        ),
        # z_representation is a bool field that _approx_eq_ compares exactly
        "DoubleFactorizedHamiltonian-z-rep": (
            ffsim.random.random_double_factorized_hamiltonian(
                norb, z_representation=True, seed=RNG
            )
        ),
        "SingleFactorizedHamiltonian": (
            ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
                mol_hamiltonian
            )
        ),
        # Variational ansatze
        "UCJOpSpinBalanced": ucj_op,
        "UCJOpSpinUnbalanced": ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
        ),
        "UCJOpSpinless": ffsim.random.random_ucj_op_spinless(
            norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
        ),
        "UCJAnglesOpSpinBalanced": ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(ucj_op),
        "GivensAnsatzOp": ffsim.GivensAnsatzOp.from_orbital_rotation(
            ffsim.random.random_unitary(norb, seed=RNG)
        ),
        "GivensAnsatzOp-no-phases": ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=[(0, 1), (2, 3)],
            thetas=RNG.uniform(-np.pi, np.pi, size=2),
            phis=None,
            phase_angles=None,
        ),
        "NumNumAnsatzOpSpinBalanced": (
            ffsim.NumNumAnsatzOpSpinBalanced.from_diag_coulomb_mats(
                np.stack(
                    [
                        ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
                        for _ in range(2)
                    ]
                )
            )
        ),
        "HopGateAnsatzOperator": ffsim.HopGateAnsatzOperator(
            norb=norb,
            interaction_pairs=[(0, 1), (2, 3)],
            thetas=RNG.uniform(-np.pi, np.pi, size=2),
        ),
        "UCCSDOpRestrictedReal": ffsim.random.random_uccsd_op_restricted_real(
            norb, nocc=2, with_final_orbital_rotation=True, seed=RNG
        ),
        "UCCSDOpRestricted": ffsim.random.random_uccsd_op_restricted(
            norb, nocc=2, with_final_orbital_rotation=True, seed=RNG
        ),
        "UCCSDOpUnrestrictedReal": ffsim.random.random_uccsd_op_unrestricted_real(
            norb, (2, 1), with_final_orbital_rotation=True, seed=RNG
        ),
        "UCCSDOpUnrestricted": ffsim.random.random_uccsd_op_unrestricted(
            norb, (2, 1), with_final_orbital_rotation=True, seed=RNG
        ),
        "UCCSDOpSpinlessReal": ffsim.random.random_uccsd_op_spinless_real(
            norb, nocc=2, with_final_orbital_rotation=True, seed=RNG
        ),
        "UCCSDOpSpinless": ffsim.random.random_uccsd_op_spinless(
            norb, nocc=2, with_final_orbital_rotation=True, seed=RNG
        ),
        # States
        "StateVector": ffsim.StateVector(vec=vec, norb=norb, nelec=(2, 2)),
        # nelec is a union of int and tuple, which JSON cannot distinguish on its own
        "StateVector-spinless": ffsim.StateVector(
            vec=ffsim.random.random_state_vector(ffsim.dim(norb, 2), seed=RNG),
            norb=norb,
            nelec=2,
        ),
        "ReducedDensityMatrix": ffsim.ReducedDensityMatrix(
            one_rdm=one_rdm, two_rdm=two_rdm
        ),
        # Operators
        "FermionOperator": ffsim.FermionOperator(
            {
                (ffsim.cre_a(0), ffsim.des_b(1)): 1.5 - 0.25j,
                (
                    ffsim.cre_b(2),
                    ffsim.cre_a(1),
                    ffsim.des_a(0),
                    ffsim.des_b(3),
                ): -0.75j,
                (): 2.0,
            }
        ),
        # Molecular data
        "MolecularData": mol_data,
        # PySCF also accepts a dictionary mapping elements to basis sets
        "MolecularData-dict-basis": dataclasses.replace(
            mol_data, basis={"H": "sto-6g"}
        ),
    }


OBJECTS = _objects()


@pytest.mark.parametrize("key", list(OBJECTS))
def test_json_dict_round_trip(key: str):
    """Test that to_json_dict and from_json_dict round-trip losslessly."""
    obj = OBJECTS[key]
    data = ffsim.to_json_dict(obj)
    assert data["__type__"].startswith("ffsim.")
    json_data = json.loads(json.dumps(data))
    assert json_data == data
    assert_exactly_equal(ffsim.from_json_dict(json_data), obj)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_json_file_round_trip(compression: str | None, tmp_path: pathlib.Path):
    """Test to_json and from_json."""
    obj = OBJECTS["UCJOpSpinBalanced"]
    path = tmp_path / "obj.json"
    ffsim.to_json(obj, path, compression=compression)
    assert_exactly_equal(ffsim.from_json(path, compression=compression), obj)


@pytest.mark.parametrize(
    "filename, compression",
    [
        ("obj.json", None),
        ("obj", None),
        ("obj.json.gz", "gzip"),
        ("obj.json.GZ", "gzip"),
        ("obj.json.bz2", "bz2"),
        ("obj.json.xz", "lzma"),
        ("obj.json.lzma", "lzma"),
    ],
)
def test_compression_inferred_from_extension(
    filename: str, compression: str | None, tmp_path: pathlib.Path
):
    """Test that to_json chooses the compression from the file extension."""
    obj = OBJECTS["UCJOpSpinBalanced"]
    path = tmp_path / filename
    ffsim.to_json(obj, path)
    assert_exactly_equal(ffsim.from_json(path, compression=compression), obj)
    assert_exactly_equal(ffsim.from_json(path), obj)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_compression_detected_from_contents(
    compression: str | None, tmp_path: pathlib.Path
):
    """Test that from_json detects the compression even if the extension is wrong."""
    obj = OBJECTS["UCJOpSpinBalanced"]
    path = tmp_path / ("obj.json.gz" if compression is None else "obj.json")
    ffsim.to_json(obj, path, compression=compression)
    assert_exactly_equal(ffsim.from_json(path), obj)


def test_explicit_compression_overrides_extension(tmp_path: pathlib.Path):
    """Test that an explicit compression argument takes precedence."""
    obj = OBJECTS["UCJOpSpinBalanced"]
    path = tmp_path / "obj.json.gz"
    ffsim.to_json(obj, path, compression=None)
    assert path.read_bytes().startswith(b"{")
    assert_exactly_equal(ffsim.from_json(path, compression=None), obj)
    path = tmp_path / "obj.json"
    ffsim.to_json(obj, path, compression="bz2")
    assert path.read_bytes().startswith(b"BZh")


def test_unsupported_compression_raises(tmp_path: pathlib.Path):
    """Test that an unknown compression algorithm raises an error."""
    with pytest.raises(ValueError, match="Unsupported compression"):
        ffsim.to_json(OBJECTS["UCJOpSpinBalanced"], tmp_path / "obj", "zip")


def test_registry_is_complete():
    """Test that every public dataclass is registered as serializable."""
    registered = set(_REGISTRY.values())
    # Look names up in the module namespace rather than with getattr, which would
    # trigger the lazy import of ffsim.qiskit and fail if Qiskit is not installed.
    unregistered = [
        name
        for name in ffsim.__all__
        if isinstance(obj := vars(ffsim).get(name), type)
        and dataclasses.is_dataclass(obj)
        and obj not in registered
    ]
    assert not unregistered, (
        f"These public dataclasses are not registered as serializable: {unregistered}. "
        "Add the @_register_serializable decorator to them."
    )


def test_numpy_values_stored_as_plain_values():
    """Test that NumPy array subclasses and scalars are stored as plain values."""
    vec = ffsim.random.random_state_vector(ffsim.dim(4, (2, 2)), seed=RNG)
    obj = SerializationValue([vec.view(FCIvector), np.float64(1.5)])
    loaded = ffsim.from_json_dict(ffsim.to_json_dict(obj)).value
    assert_exactly_equal(loaded, [vec, 1.5])


@pytest.mark.parametrize("value", [np.longdouble("1.25"), np.clongdouble(1.25 + 2.5j)])
def test_extended_precision_numpy_scalars_raise(value):
    """Test that scalars without Python equivalents raise a clear error."""
    with pytest.raises(TypeError, match="no equivalent Python scalar type"):
        ffsim.to_json_dict(SerializationValue(value))


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
def test_extended_precision_numpy_arrays_round_trip(dtype):
    """Test that rejecting extended precision scalars does not affect arrays."""
    obj = SerializationValue(np.array([1.25, 2.5], dtype=dtype))
    data = json.loads(json.dumps(ffsim.to_json_dict(obj)))
    assert_exactly_equal(ffsim.from_json_dict(data), obj)


def test_non_contiguous_arrays_round_trip():
    """Test that arrays that are not C-contiguous round-trip exactly."""
    mat = ffsim.random.random_unitary(4, seed=RNG)
    vec = RNG.standard_normal(10)
    obj = SerializationValue([np.asfortranarray(mat), vec[::2], mat.T])
    loaded = ffsim.from_json_dict(json.loads(json.dumps(ffsim.to_json_dict(obj))))
    assert_exactly_equal(loaded.value, [mat, vec[::2], mat.T])


def test_dict_round_trip():
    """Test that dictionaries with string keys round-trip exactly."""
    value = {
        "array": RNG.standard_normal(3),
        "tuple": (1, 2.5),
        "nested": {"x": [1, "a"]},
        # A key that matches the type tag key is stored as an ordinary key.
        "__type__": "not a type",
    }
    data = json.loads(json.dumps(ffsim.to_json_dict(SerializationValue(value))))
    assert_exactly_equal(ffsim.from_json_dict(data).value, value)


@pytest.mark.parametrize("value", [np.array(["a", "b"]), object(), {1: "a"}])
def test_unsupported_values_raise(value):
    """Test that unsupported values raise rather than being silently degraded."""
    with pytest.raises(TypeError, match="Could not serialize"):
        ffsim.to_json_dict(SerializationValue(value))


def test_duplicate_name_raises():
    """Test that registering a second class under a taken name raises an error."""

    @dataclasses.dataclass
    class Plain:
        x: int

    with pytest.raises(ValueError, match="already registered"):
        _register_serializable(name="test.SerializationValue")(Plain)


def test_deserialization_errors():
    """Test errors for unregistered types and unknown names."""

    @dataclasses.dataclass
    class NotRegistered:
        x: int

    with pytest.raises(TypeError, match="Only ffsim's own types"):
        ffsim.to_json_dict(NotRegistered(1))
    with pytest.raises(TypeError, match=r"ffsim\.NotAType"):
        ffsim.from_json_dict({"__type__": "ffsim.NotAType"})


def test_unvalidated_objects_round_trip():
    """Test that objects constructed with validate=False can be deserialized."""
    norb = 3
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=1, seed=RNG)
    # Scaled orbital rotations are not unitary, so validation would reject them
    obj = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=ucj_op.diag_coulomb_mats,
        orbital_rotations=1.001 * ucj_op.orbital_rotations,
        validate=False,
    )
    with pytest.raises(ValueError, match="unitary"):
        ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=obj.diag_coulomb_mats,
            orbital_rotations=obj.orbital_rotations,
        )
    data = json.loads(json.dumps(ffsim.to_json_dict(obj)))
    assert_exactly_equal(ffsim.from_json_dict(data), obj)


def test_molecular_data_methods_deprecated(tmp_path: pathlib.Path):
    """Test the deprecated MolecularData to_json and from_json methods."""
    mol_data = _molecular_data()
    path = tmp_path / "molecular_data.json.gz"
    with pytest.deprecated_call():
        mol_data.to_json(path, compression="gzip")
    with pytest.deprecated_call():
        loaded = ffsim.MolecularData.from_json(path, compression="gzip")
    assert_exactly_equal(loaded, mol_data)
    # Unlike ffsim.to_json, MolecularData.to_json does not compress by default
    path = tmp_path / "molecular_data.json.xz"
    with pytest.deprecated_call():
        mol_data.to_json(path)
    assert path.read_bytes().startswith(b"{")
    with pytest.deprecated_call():
        loaded = ffsim.MolecularData.from_json(path)
    assert_exactly_equal(loaded, mol_data)


def test_from_json_reads_legacy_molecular_data(tmp_path: pathlib.Path):
    """Test loading legacy MolecularData with untagged arrays stored as lists."""
    one_body_integrals = np.array([[1.0, 0.5], [0.5, 2.0]])
    two_body_integrals = np.arange(16.0).reshape(2, 2, 2, 2)
    legacy = {
        "core_energy": 0.25,
        "one_body_integrals": one_body_integrals.tolist(),
        "two_body_integrals": two_body_integrals.tolist(),
        "norb": 2,
        "nelec": [1, 1],
        "atom": [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.75]]],
        "basis": "sto-3g",
        "hf_energy": -1.0,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    gzip_path = tmp_path / "legacy.json.gz"
    gzip_path.write_bytes(gzip.compress(json.dumps(legacy).encode()))

    for loaded in [
        ffsim.from_json(path),
        ffsim.from_json(gzip_path, compression="gzip"),
        ffsim.from_json(gzip_path),
    ]:
        assert loaded.core_energy == 0.25
        assert loaded.norb == 2
        assert loaded.nelec == (1, 1)
        assert loaded.basis == "sto-3g"
        assert loaded.hf_energy == -1.0
        assert loaded.atom == [("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.75])]
        np.testing.assert_allclose(loaded.one_body_integrals, one_body_integrals)
        np.testing.assert_allclose(loaded.two_body_integrals, two_body_integrals)
