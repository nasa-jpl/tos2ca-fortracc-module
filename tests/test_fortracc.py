import numpy as np
import pytest

from fortracc_module.detectors import LessThanDetector, GreaterThanDetector
from fortracc_module.fortracc import FortraccRunner
from fortracc_module.objects import GeoGrid


def make_grid(rows=20, cols=20):
    return GeoGrid(
        list(np.linspace(0, 10, rows)),
        list(np.linspace(0, 10, cols)),
    )


def make_single_blob_images(rows=20, cols=20, n_times=3):
    """Images with one persistent cold blob (value=200) on a warm background (value=300)."""
    images = []
    for _ in range(n_times):
        img = np.full((rows, cols), 300.0)
        img[5:15, 5:15] = 200.0
        images.append(img)
    return images


def make_timestamps(n):
    base = 201501010000
    return [str(base + i * 100) for i in range(n)]


class TestFortraccRunnerInit:
    def test_mismatched_lengths_raises(self):
        grid = make_grid()
        images = make_single_blob_images(n_times=3)
        with pytest.raises(ValueError, match="number of images"):
            FortraccRunner(images, ['201501010000', '201501010100'], grid, LessThanDetector(250.0))

    def test_bad_timestamp_length_raises(self):
        grid = make_grid()
        images = make_single_blob_images(n_times=1)
        with pytest.raises(ValueError, match="YYYYMMDDhhmm"):
            FortraccRunner(images, ['20150101'], grid, LessThanDetector(250.0))

    def test_non_2d_image_raises(self):
        grid = make_grid()
        images = [np.ones((20, 20, 3))]
        with pytest.raises(ValueError, match="2D"):
            FortraccRunner(images, ['201501010000'], grid, LessThanDetector(250.0))

    def test_stores_attributes(self):
        grid = make_grid()
        images = make_single_blob_images(n_times=2)
        timestamps = make_timestamps(2)
        detector = LessThanDetector(250.0)
        runner = FortraccRunner(images, timestamps, grid, detector, min_size=1)
        assert runner.images is images
        assert runner.timestamps == timestamps
        assert runner.grid is grid
        assert runner.detector is detector


class TestFortraccRunnerTimeSeries:
    def test_single_persistent_event(self):
        """One blob that persists across all time steps → one time series."""
        grid = make_grid()
        images = make_single_blob_images(n_times=4)
        timestamps = make_timestamps(4)
        runner = FortraccRunner(images, timestamps, grid, LessThanDetector(250.0), min_size=1)
        assert len(runner.time_series) == 1

    def test_single_event_continuation_tags(self):
        """A blob appearing in 3 frames should have initiation, continuation, continuation tags."""
        grid = make_grid()
        images = make_single_blob_images(n_times=3)
        timestamps = make_timestamps(3)
        runner = FortraccRunner(images, timestamps, grid, LessThanDetector(250.0), min_size=1)
        series = runner.time_series[0]
        tags = [tag for _, _, tag in series]
        assert tags[0] == 'initiation'
        assert all(t == 'continuation' for t in tags[1:])

    def test_no_events_when_threshold_never_met(self):
        grid = make_grid()
        images = [np.full((20, 20), 300.0) for _ in range(3)]
        timestamps = make_timestamps(3)
        runner = FortraccRunner(images, timestamps, grid, LessThanDetector(100.0), min_size=1)
        assert len(runner.time_series) == 0

    def test_initiation_and_dissipation(self):
        """Blob present in first frame only → one series; sole entry gets dissipation tag."""
        grid = make_grid()
        img_with_blob = np.full((20, 20), 300.0)
        img_with_blob[5:15, 5:15] = 200.0
        img_empty = np.full((20, 20), 300.0)
        images = [img_with_blob, img_empty]
        timestamps = make_timestamps(2)
        runner = FortraccRunner(images, timestamps, grid, LessThanDetector(250.0), min_size=1)
        assert len(runner.time_series) == 1
        tags = [tag for _, _, tag in runner.time_series[0]]
        # _add_dissipation overwrites the last tag, so a single-frame blob ends up 'dissipation'
        assert 'dissipation' in tags[-1]

    def test_two_independent_events(self):
        """Two spatially separate blobs, neither overlapping → two time series."""
        grid = make_grid(30, 30)
        images = []
        for _ in range(2):
            img = np.full((30, 30), 300.0)
            img[2:8, 2:8] = 200.0
            img[22:28, 22:28] = 200.0
            images.append(img)
        timestamps = make_timestamps(2)
        runner = FortraccRunner(
            images, timestamps, GeoGrid(list(np.linspace(0, 10, 30)), list(np.linspace(0, 10, 30))),
            LessThanDetector(250.0), min_size=1
        )
        assert len(runner.time_series) == 2

    def test_masks_computed_from_detector(self):
        grid = make_grid()
        images = make_single_blob_images(n_times=2)
        timestamps = make_timestamps(2)
        detector = LessThanDetector(250.0)
        runner = FortraccRunner(images, timestamps, grid, detector, min_size=1)
        expected_masks = detector.create_masks(images)
        for got, expected in zip(runner.masks, expected_masks):
            np.testing.assert_array_equal(got, expected)
