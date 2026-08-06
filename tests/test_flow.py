import numpy as np
import pytest

from fortracc_module.detectors import LessThanDetector
from fortracc_module.flow import SparseTimeOrderedSequence
from fortracc_module.objects import GeoGrid, SparseGeoGrid, SparseMask


def make_grid(rows=20, cols=20):
    return GeoGrid(
        list(np.linspace(0, 10, rows)),
        list(np.linspace(0, 10, cols)),
    )


def make_timestamps(n, base=201501010000):
    return [str(base + i * 100) for i in range(n)]


def make_blob_images(rows=20, cols=20, n_times=3):
    images = []
    for _ in range(n_times):
        img = np.full((rows, cols), 300.0)
        img[5:15, 5:15] = 200.0
        images.append(img)
    return images


def run_stos(n_times=3, rows=20, cols=20):
    images = make_blob_images(rows, cols, n_times)
    timestamps = make_timestamps(n_times)
    grid = make_grid(rows, cols)
    return SparseTimeOrderedSequence.run_fortracc(
        images, timestamps, grid, LessThanDetector(250.0), min_size=1
    )


class TestSparseTimeOrderedSequenceRunFortracc:
    def test_returns_stos_instance(self):
        stos = run_stos()
        assert isinstance(stos, SparseTimeOrderedSequence)

    def test_one_persistent_event(self):
        stos = run_stos(n_times=3)
        assert len(stos.events) == 1

    def test_timestamps_stored(self):
        n = 4
        stos = run_stos(n_times=n)
        assert len(stos.timestamps) == n

    def test_detector_type_stored(self):
        stos = run_stos()
        assert stos.detector_type == "less_than_threshold"

    def test_sparse_masks_have_correct_timestamp(self):
        timestamps = make_timestamps(3)
        images = make_blob_images(n_times=3)
        grid = make_grid()
        stos = SparseTimeOrderedSequence.run_fortracc(
            images, timestamps, grid, LessThanDetector(250.0), min_size=1
        )
        for event in stos.events:
            for sparse_mask in event:
                assert sparse_mask.timestamp in timestamps

    def test_retain_run_params(self):
        images = make_blob_images()
        timestamps = make_timestamps(3)
        grid = make_grid()
        stos = SparseTimeOrderedSequence.run_fortracc(
            images, timestamps, grid, LessThanDetector(250.0),
            min_size=1, retain_run_params=True
        )
        assert stos.fortracc_runner is not None

    def test_no_retain_run_params(self):
        stos = run_stos()
        assert stos.fortracc_runner is None


class TestSparseTimeOrderedSequenceMasks:
    def test_masks_shape(self):
        stos = run_stos(n_times=3, rows=20, cols=20)
        for event_masks in stos.masks:
            for ts, mask in event_masks.items():
                assert mask.shape == (20, 20)

    def test_masks_keys_match_timestamps(self):
        stos = run_stos(n_times=3)
        for event_masks in stos.masks:
            assert set(event_masks.keys()) == set(stos.timestamps)

    def test_data_values_shape(self):
        stos = run_stos(n_times=3, rows=20, cols=20)
        for event_values in stos.data_values:
            for ts, vals in event_values.items():
                assert vals.shape == (20, 20)


class TestSparseTimeOrderedSequenceInit:
    def _make_sparse_mask(self, timestamp='201501010000'):
        return SparseMask.from_coords(
            x_coords=[1, 2], y_coords=[1, 2],
            values=[200.0, 200.0],
            timestamp=timestamp,
            mask_type='initiation'
        )

    def test_stores_events(self):
        grid = SparseGeoGrid.from_lat_lon([0, 5, 10], [0, 5, 10])
        events = [[self._make_sparse_mask()]]
        stos = SparseTimeOrderedSequence(events, ['201501010000'], grid)
        assert stos.events is events

    def test_empty_detector_type_defaults_to_empty_string(self):
        grid = SparseGeoGrid.from_lat_lon([0, 5, 10], [0, 5, 10])
        stos = SparseTimeOrderedSequence([], ['201501010000'], grid)
        assert stos.detector_type == ""
