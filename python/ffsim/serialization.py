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

"""Serialization.

Functions for converting ffsim objects to and from dictionaries and JSON files.
"""

from __future__ import annotations

import base64
import bz2
import dataclasses
import gzip
import inspect
import json
import lzma
import os
import pathlib
from typing import IO, Any, Callable, TypeVar

import numpy as np

from ffsim._lib import FermionOperator

_T = TypeVar("_T", bound=type)


@dataclasses.dataclass(frozen=True)
class _Entry:
    name: str
    encode: Callable[[Any], dict[str, Any]]
    decode: Callable[[dict[str, Any]], Any]


# Registered classes, by serialized name and by class.
_REGISTRY: dict[str, type] = {}
_ENTRIES: dict[type, _Entry] = {}


def _register_serializable(*, name: str) -> Callable[[_T], _T]:
    """Register an ffsim dataclass as serializable by :func:`to_json_dict` and
    :func:`from_json_dict`.

    This is for ffsim's own classes only and is not part of the public API.

    The dataclass is serialized as its ``init`` fields and reconstructed by calling its
    constructor. If the constructor accepts a ``validate`` argument, it is passed
    ``validate=False``. Place this decorator above ``@dataclasses.dataclass``.

    Args:
        name: The name to store in the serialized representation. It is part of the
            file format, so changing it makes existing files unreadable.

    Returns:
        A decorator that registers and returns the class unmodified.
    """

    def decorator(cls: _T) -> _T:
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls} must be a dataclass.")
        # Validation arguments are init-only, so they are not serialized. Validating
        # again on load would use the default tolerances, which would reject an
        # object that was constructed with validate=False or looser tolerances.
        has_validate = "validate" in inspect.signature(cls).parameters
        kwargs = {"validate": False} if has_validate else {}

        def decode(data: dict[str, Any]) -> Any:
            return cls(**data, **kwargs)

        _register(cls, _Entry(name, _encode_dataclass, decode))
        return cls

    return decorator


def _register(cls: type, entry: _Entry) -> None:
    registered = _REGISTRY.get(entry.name)
    if registered is not None and registered is not cls:
        raise ValueError(
            f"Cannot register {cls} under the name {entry.name!r} because "
            f"{registered} is already registered under that name."
        )
    _REGISTRY[entry.name] = cls
    _ENTRIES[cls] = entry


def _encode_dataclass(obj: Any) -> dict[str, Any]:
    return {
        field.name: getattr(obj, field.name)
        for field in dataclasses.fields(obj)
        if field.init
    }


def _encode_ndarray(val: np.ndarray) -> dict[str, Any]:
    if val.dtype.kind not in "biufc":
        raise TypeError(f"Could not serialize a NumPy array with dtype {val.dtype!r}.")
    return {
        "__type__": "ndarray",
        # dtype.str includes the byte order.
        "dtype": val.dtype.str,
        "shape": list(val.shape),
        # tobytes returns the data in C order, whatever the memory layout.
        "data": base64.b64encode(val.tobytes()).decode("ascii"),
    }


def _decode_ndarray(data: dict[str, Any]) -> np.ndarray:
    buffer = base64.b64decode(data["data"])
    array = np.frombuffer(buffer, dtype=np.dtype(data["dtype"]))
    # frombuffer returns a read-only view of the decoded bytes, so make a copy.
    return array.reshape(data["shape"]).copy()


def _encode_fermion_operator(val: FermionOperator) -> dict[str, Any]:
    # The keys are tuples, which cannot be used as JSON object keys, so the
    # coefficients are stored as a list of key-value pairs.
    return {
        "coeffs": [
            [[list(action) for action in term], coeff] for term, coeff in val.items()
        ]
    }


def _decode_fermion_operator(data: dict[str, Any]) -> FermionOperator:
    return FermionOperator(
        {
            tuple(
                (bool(action), bool(spin), int(orb)) for action, spin, orb in term
            ): coeff
            for term, coeff in data["coeffs"]
        }
    )


def _encode_value(val: Any) -> Any:
    """Recursively encode a field value into a JSON-compatible representation.

    Supported values are ``None``, Booleans, integers, floats, complex numbers,
    strings, lists, tuples, dictionaries with string keys, NumPy arrays and scalars
    with Boolean or numeric dtypes, and serializable ffsim objects. NumPy array
    subclasses are stored as plain arrays, and NumPy scalars are stored as Python
    scalars. NumPy scalars without an equivalent Python scalar type, such as
    ``np.longdouble`` and ``np.clongdouble``, are unsupported.
    """
    if val is None or type(val) in (bool, int, float, str):
        return val
    if isinstance(val, np.generic):
        scalar = val.item()
        if isinstance(scalar, np.generic):
            raise TypeError(
                f"Could not serialize a NumPy scalar with dtype {val.dtype!r}: "
                "there is no equivalent Python scalar type."
            )
        return _encode_value(scalar)
    # Subclasses, such as PySCF's FCIvector, are stored as plain arrays.
    if isinstance(val, np.ndarray):
        return _encode_ndarray(val)
    if type(val) is complex:
        return {"__type__": "complex", "real": val.real, "imag": val.imag}
    # Exact type checks keep subclasses, such as named tuples, from being silently
    # encoded as their base type.
    if type(val) is tuple:
        return {"__type__": "tuple", "values": [_encode_value(v) for v in val]}
    if type(val) is list:
        return [_encode_value(v) for v in val]
    if type(val) is dict:
        for key in val:
            # JSON object keys are strings, and json.dumps would silently convert
            # other keys, such as integers, to strings.
            if type(key) is not str:
                raise TypeError(
                    f"Could not serialize a dictionary with a key of type {type(key)}: "
                    f"{key!r}\nOnly dictionaries with string keys are supported."
                )
        return {
            "__type__": "dict",
            "items": {key: _encode_value(v) for key, v in val.items()},
        }
    if type(val) in _ENTRIES:
        return to_json_dict(val)
    raise TypeError(
        f"Could not serialize object of type {type(val)}: {val!r}\n"
        "The object was neither a supported primitive type nor a serializable "
        "ffsim type."
    )


def _decode_value(val: Any) -> Any:
    """Recursively decode a value produced by ``_encode_value``."""
    if type(val) is list:
        return [_decode_value(v) for v in val]
    if type(val) is not dict:
        return val
    type_name = val["__type__"]
    if type_name == "ndarray":
        return _decode_ndarray(val)
    if type_name == "complex":
        return complex(val["real"], val["imag"])
    if type_name == "tuple":
        return tuple(_decode_value(v) for v in val["values"])
    if type_name == "dict":
        return {key: _decode_value(v) for key, v in val["items"].items()}
    return from_json_dict(val)


def to_json_dict(obj: Any) -> dict[str, Any]:
    """Convert an object to a JSON-compatible dictionary.

    Supported objects are instances of ffsim's public dataclasses, such as the
    Hamiltonian, ansatz operator, :class:`StateVector`, and :class:`MolecularData`
    classes, as well as :class:`FermionOperator`. The dictionary stores the type's
    name in ``__type__`` alongside the object's encoded fields.

    Args:
        obj: The object to convert.

    Returns:
        The dictionary representation of the object.
    """
    entry = _ENTRIES.get(type(obj))
    if entry is None:
        raise TypeError(
            f"Could not serialize object of type {type(obj)}.\n"
            "Only ffsim's own types can be serialized."
        )
    return {
        "__type__": entry.name,
        **{key: _encode_value(val) for key, val in entry.encode(obj).items()},
    }


def from_json_dict(data: dict[str, Any]) -> Any:
    """Reconstruct an object from the dictionary representation produced by
    :func:`to_json_dict`.

    Args:
        data: The dictionary representation of the object.

    Returns:
        The reconstructed object.
    """
    if "__type__" not in data:
        raise TypeError("Could not deserialize a dictionary with no __type__ key.")
    type_name = data["__type__"]
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise TypeError(
            f"Could not deserialize an object of type {type_name!r}.\n"
            "This version of ffsim does not recognize that type name. If the data was "
            "written by a newer version of ffsim, you may need to upgrade."
        )
    fields = {key: _decode_value(val) for key, val in data.items() if key != "__type__"}
    return _ENTRIES[cls].decode(fields)


_OPEN_FUNCS: dict[str | None, Callable] = {
    None: open,
    "gzip": gzip.open,
    "bz2": bz2.open,
    "lzma": lzma.open,
}


# Signatures at the start of files written by gzip.open, bz2.open, and lzma.open.
# Valid JSON cannot start with any of them.
_MAGIC_BYTES = {
    b"\x1f\x8b": "gzip",
    b"BZh": "bz2",
    b"\xfd7zXZ\x00": "lzma",
}

_EXTENSIONS = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "lzma",
    ".lzma": "lzma",
}


def _detect_compression(file: str | bytes | os.PathLike) -> str | None:
    with open(file, "rb") as f:
        head = f.read(max(len(magic) for magic in _MAGIC_BYTES))
    for magic, compression in _MAGIC_BYTES.items():
        if head.startswith(magic):
            return compression
    return None


def _open(
    file: str | bytes | os.PathLike, mode: str, compression: str | None
) -> IO[bytes]:
    if compression == "infer":
        if "r" in mode:
            # The file contents are more reliable than its name.
            compression = _detect_compression(file)
        else:
            suffix = pathlib.Path(os.fsdecode(file)).suffix.lower()
            compression = _EXTENSIONS.get(suffix)
    if compression not in _OPEN_FUNCS:
        raise ValueError(
            f"Unsupported compression algorithm: {compression!r}. Supported options "
            'are "gzip", "bz2", "lzma", "infer", or None for no compression.'
        )
    return _OPEN_FUNCS[compression](file, mode)


def to_json(
    obj: Any, file: str | bytes | os.PathLike, compression: str | None = "infer"
) -> None:
    """Serialize an object to a (possibly compressed) JSON file.

    See :func:`to_json_dict` for supported objects.

    Args:
        obj: The object to serialize.
        file: The file path to write to.
        compression: The compression algorithm to use: ``"gzip"``, ``"bz2"``,
            ``"lzma"``, or ``None`` for no compression. The default, ``"infer"``,
            chooses it from the file extension: ``.gz`` for gzip, ``.bz2`` for bz2,
            ``.xz`` or ``.lzma`` for lzma, and no compression for any other
            extension.
    """
    payload = json.dumps(to_json_dict(obj), separators=(",", ":"))
    with _open(file, "wb", compression) as f:
        f.write(payload.encode())


def from_json(
    file: str | bytes | os.PathLike, compression: str | None = "infer"
) -> Any:
    """Load an object from a (possibly compressed) JSON file.

    Files written by :meth:`MolecularData.to_json` in ffsim 0.0.86 and earlier are
    also supported.

    Args:
        file: The file path to read from.
        compression: The compression algorithm that was used to compress the file:
            ``"gzip"``, ``"bz2"``, ``"lzma"``, or ``None`` for no compression. The
            default, ``"infer"``, detects it from the file's contents.

    Returns:
        The deserialized object.
    """
    with _open(file, "rb", compression) as f:
        data = json.load(f)
    if "__type__" not in data:
        # ffsim 0.0.86 and earlier wrote MolecularData objects with no type tag, and
        # did not write any other type to JSON.
        from ffsim.molecular_data import MolecularData

        return MolecularData._from_json_legacy_(data)
    return from_json_dict(data)


_register(
    FermionOperator,
    _Entry("ffsim.FermionOperator", _encode_fermion_operator, _decode_fermion_operator),
)
