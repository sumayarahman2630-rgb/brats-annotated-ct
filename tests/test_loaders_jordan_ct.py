"""Tests for the Jordan hospital DICOM loader: filename-based CT/mask
matching (and flagging of unmatched files), RGB DICOM reading, grayscale
conversion, and normalization/binarization. CPU-only, uses pydicom to
build real (if minimal) DICOM fixtures rather than mocking the reader.
"""
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from data.loaders_jordan_ct import (
    JordanCTSegDataset,
    _binarize_jordan_mask,
    _normalize_jordan_ct,
    _read_dicom_grayscale,
    discover_jordan_slices,
)


def _write_rgb_dicom(path, pixel_array):
    """Writes a minimal, valid RGB secondary-capture DICOM file -- mirrors
    the real Jordan export format (RGB, 0-255, windowed) described in
    PROJECT_NOTES.md's Stage 3 section."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = pixel_array.shape[0], pixel_array.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), enforce_file_format=True)


def _write_fake_jordan(tmp_path, matched=((("P001", 1)), (("P001", 2))), ct_only=(("P001", 3),), mask_only=()):
    """Builds a fake ct_root/mask_root pair with the given matched,
    CT-only, and mask-only (patient_id, slice_num) keys."""
    ct_root = tmp_path / "annoted-20-ct"
    mask_root = tmp_path / "mask-data-20"
    ct_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for pid, slice_num in matched:
        ct_arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mask_arr = np.zeros((32, 32, 3), dtype=np.uint8)
        mask_arr[10:20, 10:20, :] = 255
        _write_rgb_dicom(ct_root / f"{pid}_CT_s{slice_num}.dcm", ct_arr)
        _write_rgb_dicom(mask_root / f"{pid}_CT_m{slice_num}.dcm", mask_arr)

    for pid, slice_num in ct_only:
        _write_rgb_dicom(ct_root / f"{pid}_CT_s{slice_num}.dcm", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))

    for pid, slice_num in mask_only:
        _write_rgb_dicom(mask_root / f"{pid}_CT_m{slice_num}.dcm", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))

    return str(ct_root), str(mask_root)


def test_discover_jordan_slices_matches_ct_and_mask_by_key(tmp_path):
    ct_root, mask_root = _write_fake_jordan(tmp_path)
    slices = discover_jordan_slices(ct_root, mask_root)
    keys = {(s.patient_id, s.slice_num) for s in slices}
    assert keys == {("P001", 1), ("P001", 2)}, f"unexpected matched keys: {keys}"


def test_discover_jordan_slices_excludes_unmatched_ct_only_slice(tmp_path):
    """The P001_CT_s3.dcm fixture has no matching mask -- must be excluded,
    not paired with the wrong mask or crash."""
    ct_root, mask_root = _write_fake_jordan(tmp_path, ct_only=(("P001", 3),))
    slices = discover_jordan_slices(ct_root, mask_root)
    assert all(s.slice_num != 3 for s in slices)
    assert len(slices) == 2


def test_discover_jordan_slices_excludes_unmatched_mask_only_slice(tmp_path):
    ct_root, mask_root = _write_fake_jordan(tmp_path, ct_only=(), mask_only=(("P002", 1),))
    slices = discover_jordan_slices(ct_root, mask_root)
    assert all(s.patient_id != "P002" for s in slices)


def test_read_dicom_grayscale_converts_rgb_correctly(tmp_path):
    path = tmp_path / "test.dcm"
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 100  # pure red
    _write_rgb_dicom(path, rgb)
    gray = _read_dicom_grayscale(str(path))
    assert gray.ndim == 2
    expected = 0.299 * 100
    assert np.allclose(gray, expected, atol=0.01)


def test_normalize_jordan_ct_maps_to_minus_one_to_one():
    gray = np.array([[0.0, 128.0], [255.0, 64.0]], dtype=np.float32)
    normed = _normalize_jordan_ct(gray)
    assert np.isclose(normed.min(), -1.0)
    assert np.isclose(normed.max(), 1.0)


def test_binarize_jordan_mask_thresholds_correctly():
    gray = np.array([[0.0, 200.0], [50.0, 255.0]], dtype=np.float32)
    binary = _binarize_jordan_mask(gray, threshold=127.0)
    assert np.array_equal(binary, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))


def test_jordan_dataset_returns_normalized_binary_tensors(tmp_path):
    ct_root, mask_root = _write_fake_jordan(tmp_path)
    slices = discover_jordan_slices(ct_root, mask_root)
    ds = JordanCTSegDataset(slices)
    item = ds[0]
    assert item["ct"].shape[0] == 1  # (1, H, W)
    assert item["mask"].shape[0] == 1
    assert set(item["mask"].unique().tolist()).issubset({0.0, 1.0})
    assert item["ct"].min() >= -1.0 and item["ct"].max() <= 1.0
