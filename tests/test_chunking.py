import numpy as np
import pytest
import warnings

from fortracc_module.chunking import create_file_chunks, stitch
from fortracc_module.detectors import LessThanDetector
from fortracc_module.flow import SparseTimeOrderedSequence
from fortracc_module.objects import GeoGrid


def make_timestamps_tuples(n, base=201501010000):
    return [(str(base + i * 100), f'file_{i}.nc') for i in range(n)]


def make_grid(rows=20, cols=20):
    return GeoGrid(
        list(np.linspace(0, 10, rows)),
        list(np.linspace(0, 10, cols)),
    )


def make_blob_images(rows=20, cols=20, n_times=3):
    images = []
    for _ in range(n_times):
        img = np.full((rows, cols), 300.0)
        img[5:15, 5:15] = 200.0
        images.append(img)
    return images


def run_stos(timestamps_list, rows=20, cols=20):
    n = len(timestamps_list)
    images = make_blob_images(rows, cols, n)
    grid = make_grid(rows, cols)
    return SparseTimeOrderedSequence.run_fortracc(
        images, timestamps_list, grid, LessThanDetector(250.0), min_size=1
    )


class TestCreateFileChunks:
    def test_fewer_than_chunk_size_returns_flat(self):
        fns = make_timestamps_tuples(5)
        result = create_file_chunks(fns, num_per_chunk=20)
        assert result == sorted(fns, key=lambda x: x[0])

    def test_sorted_by_timestamp(self):
        fns = list(reversed(make_timestamps_tuples(5)))
        result = create_file_chunks(fns, num_per_chunk=20)
        timestamps = [r[0] for r in result]
        assert timestamps == sorted(timestamps)

    def test_chunked_output_is_list_of_lists(self):
        fns = make_timestamps_tuples(25)
        result = create_file_chunks(fns, num_per_chunk=10)
        assert isinstance(result, list)
        assert all(isinstance(chunk, list) for chunk in result)

    def test_chunks_overlap_by_one(self):
        fns = make_timestamps_tuples(25)
        result = create_file_chunks(fns, num_per_chunk=10)
        for i in range(len(result) - 1):
            assert result[i][-1] == result[i + 1][0]

    def test_exact_multiple_chunk_size(self):
        fns = make_timestamps_tuples(20)
        result = create_file_chunks(fns, num_per_chunk=10)
        assert isinstance(result, list)
        assert all(isinstance(chunk, list) for chunk in result)


class TestStitch:
    def _make_stos_pair(self):
        """Two overlapping STOS objects sharing one timestamp."""
        ts1 = [str(201501010000 + i * 100) for i in range(4)]
        ts2 = ts1[3:4] + [str(201501010000 + i * 100) for i in range(4, 7)]

        stos1 = run_stos(ts1)
        stos2 = run_stos(ts2)
        return stos1, stos2

    def test_stitch_returns_stos(self):
        stos1, stos2 = self._make_stos_pair()
        result = stitch([stos1, stos2])
        assert isinstance(result, SparseTimeOrderedSequence)

    def test_stitch_timestamps_merged(self):
        stos1, stos2 = self._make_stos_pair()
        result = stitch([stos1, stos2])
        # stitch() includes the shared timestamp once in stos1 then appends all of stos2:
        # 4 from first + 4 from second (stitch extends with timestamps[1:]) = 7
        assert len(result.timestamps) == 7

    def test_stitch_single_item(self):
        stos1, _ = self._make_stos_pair()
        result = stitch([stos1])
        assert result.timestamps == stos1.timestamps

    def test_stitch_mismatched_grids_raises(self):
        ts1 = [str(201501010000 + i * 100) for i in range(2)]
        ts2 = ts1[1:2] + [str(201501010200)]

        grid1 = make_grid(20, 20)
        grid2 = make_grid(30, 30)

        images1 = make_blob_images(20, 20, 2)
        images2 = make_blob_images(30, 30, 2)

        stos1 = SparseTimeOrderedSequence.run_fortracc(
            images1, ts1, grid1, LessThanDetector(250.0), min_size=1
        )
        stos2 = SparseTimeOrderedSequence.run_fortracc(
            images2, ts2, grid2, LessThanDetector(250.0), min_size=1
        )
        with pytest.raises(ValueError, match="same grid"):
            stitch([stos1, stos2])

    def test_stitch_non_overlapping_raises(self):
        ts1 = [str(201501010000 + i * 100) for i in range(2)]
        ts2 = [str(201501010300 + i * 100) for i in range(2)]  # gap, no overlap

        stos1 = run_stos(ts1)
        stos2 = run_stos(ts2)
        with pytest.raises(ValueError, match="overlapping"):
            stitch([stos1, stos2])

    def test_stitch_mismatched_detector_raises(self):
        ts1 = [str(201501010000 + i * 100) for i in range(2)]
        ts2 = ts1[1:2] + [str(201501010200)]

        grid = make_grid()
        images1 = make_blob_images(n_times=2)
        images2 = make_blob_images(n_times=2)

        stos1 = SparseTimeOrderedSequence.run_fortracc(
            images1, ts1, grid, LessThanDetector(250.0), min_size=1
        )
        stos2 = SparseTimeOrderedSequence.run_fortracc(
            images2, ts2, grid, LessThanDetector(250.0), min_size=1
        )
        stos2.detector_type = "different_detector"

        with pytest.raises(ValueError, match="detector_type"):
            stitch([stos1, stos2])
