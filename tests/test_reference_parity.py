import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from streamner_commit.reference.parity import (
    ARRAYS_FILENAME,
    METADATA_FILENAME,
    REQUIRED_ARRAY_NAMES,
    ParityFixture,
    deterministic_npz_bytes,
    validate_arrays,
    write_parity_fixture,
)


def test_deterministic_npz_is_byte_stable_sorted_and_pickle_free() -> None:
    arrays = {
        "z_values": np.array([[1.5, -2.0]], dtype=np.float32),
        "a_mask": np.array([[True, False]], dtype=np.bool_),
        "indices": np.array([3, 1], dtype=np.int64),
    }
    first = deterministic_npz_bytes(arrays)
    second = deterministic_npz_bytes(dict(reversed(tuple(arrays.items()))))

    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a_mask.npy", "indices.npy", "z_values.npy"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)


@pytest.mark.parametrize(
    ("arrays", "match"),
    [
        ({"bad/name": np.array([1])}, "unsafe parity array name"),
        ({"objects": np.array([object()], dtype=object)}, "unsafe or non-numeric"),
        ({"values": np.array([np.inf])}, "non-finite"),
    ],
)
def test_validate_arrays_rejects_unsafe_payloads(
    arrays: dict[str, np.ndarray],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_arrays(arrays)


def test_validate_arrays_requires_named_capture_members() -> None:
    with pytest.raises(ValueError, match="missing required arrays: expected"):
        validate_arrays(
            {"actual": np.array([1], dtype=np.int64)},
            required=frozenset({"expected"}),
        )


def test_writer_adds_stable_manifest_and_checksums(tmp_path) -> None:
    arrays = {
        name: np.array([index], dtype=np.int64)
        for index, name in enumerate(sorted(REQUIRED_ARRAY_NAMES))
    }
    arrays["value"] = np.array([[1.0, 2.0]], dtype=np.float32)
    fixture = ParityFixture(arrays=arrays, metadata={"model_revision_sha": "a" * 40})

    first = write_parity_fixture(fixture, tmp_path)
    first_npz = (tmp_path / ARRAYS_FILENAME).read_bytes()
    first_json = (tmp_path / METADATA_FILENAME).read_bytes()
    second = write_parity_fixture(fixture, tmp_path)

    assert first_npz == (tmp_path / ARRAYS_FILENAME).read_bytes()
    assert first_json == (tmp_path / METADATA_FILENAME).read_bytes()
    assert first["arrays_sha256"] == second["arrays_sha256"]
    assert first["metadata_sha256"] == second["metadata_sha256"]

    metadata = json.loads(first_json)
    assert metadata["arrays_file"]["sha256"] == hashlib.sha256(first_npz).hexdigest()
    assert metadata["arrays"]["value"] == {
        "dtype": "float32",
        "nbytes": 8,
        "numel": 2,
        "sha256": metadata["arrays"]["value"]["sha256"],
        "shape": [1, 2],
    }
    with np.load(tmp_path / ARRAYS_FILENAME, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["value"], arrays["value"])
