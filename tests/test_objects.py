import numpy as np
import pytest

from fortracc_module.objects import GeoGrid, SparseGeoGrid, Scene, SparseMask


class TestGeoGrid:
    def setup_method(self):
        self.lat = [10.0, 20.0, 30.0]
        self.lon = [-100.0, -90.0, -80.0]
        self.grid = GeoGrid(self.lat, self.lon)

    def test_lat_lon_stored(self):
        assert self.grid.latitude == self.lat
        assert self.grid.longitude == self.lon

    def test_bounds(self):
        assert self.grid.lat_bounds == (10.0, 30.0)
        assert self.grid.lon_bounds == (-100.0, -80.0)

    def test_shape(self):
        assert self.grid.shape == (3, 3)

    def test_equality_same(self):
        other = GeoGrid(self.lat, self.lon)
        assert self.grid == other

    def test_equality_different_bounds(self):
        other = GeoGrid([0.0, 10.0, 20.0], self.lon)
        assert not (self.grid == other)

    def test_equality_type_error(self):
        with pytest.raises(ValueError):
            self.grid == "not a grid"


class TestSparseGeoGrid:
    def setup_method(self):
        self.lat = [10.0, 20.0, 30.0]
        self.lon = [-100.0, -90.0, -80.0]
        self.grid = GeoGrid(self.lat, self.lon)

    def test_from_grid(self):
        sparse = SparseGeoGrid.from_grid(self.grid)
        assert sparse.lat_bounds == self.grid.lat_bounds
        assert sparse.lon_bounds == self.grid.lon_bounds
        assert sparse.shape == self.grid.shape

    def test_from_lat_lon(self):
        sparse = SparseGeoGrid.from_lat_lon(self.lat, self.lon)
        assert sparse.shape == (3, 3)
        assert sparse.lat_bounds == (10.0, 30.0)
        assert sparse.lon_bounds == (-100.0, -80.0)

    def test_latitude_property_linspace(self):
        sparse = SparseGeoGrid.from_lat_lon(self.lat, self.lon)
        lats = sparse.latitude
        assert len(lats) == 3
        assert lats[0] == pytest.approx(10.0)
        assert lats[-1] == pytest.approx(30.0)

    def test_longitude_property_linspace(self):
        sparse = SparseGeoGrid.from_lat_lon(self.lat, self.lon)
        lons = sparse.longitude
        assert len(lons) == 3
        assert lons[0] == pytest.approx(-100.0)
        assert lons[-1] == pytest.approx(-80.0)


class TestScene:
    def _simple_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        return mask

    def test_single_component(self):
        mask = self._simple_mask()
        scene = Scene(mask, timestamp='201501011200')
        assert len(scene.events) == 1

    def test_two_components(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True
        mask[7:9, 7:9] = True
        scene = Scene(mask, timestamp='201501011200')
        assert len(scene.events) == 2

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        scene = Scene(mask, timestamp='201501011200')
        assert len(scene.events) == 0

    def test_min_size_filter(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True   # 4 pixels
        mask[8, 8] = True       # 1 pixel — filtered out by min_size=2
        scene = Scene(mask, timestamp='201501011200', min_size=2)
        assert len(scene.events) == 1

    def test_event_id_format(self):
        mask = self._simple_mask()
        scene = Scene(mask, timestamp='201501011200')
        event_id = list(scene.events.keys())[0]
        assert event_id.startswith('201501011200.')

    def test_labels_shape(self):
        mask = self._simple_mask()
        scene = Scene(mask, timestamp='201501011200')
        assert scene.labels.shape == mask.shape

    def test_getitem(self):
        mask = self._simple_mask()
        scene = Scene(mask, timestamp='201501011200')
        event_id = list(scene.events.keys())[0]
        assert scene[event_id] is scene.events[event_id]


class TestSparseMask:
    def test_from_coords_bbox(self):
        sm = SparseMask.from_coords(
            x_coords=[1, 2, 3],
            y_coords=[4, 5, 6],
            values=[1.0, 2.0, 3.0],
            timestamp='201501011200',
            mask_type='initiation',
        )
        assert sm.bbox == (4, 1, 7, 4)

    def test_from_coords_empty(self):
        sm = SparseMask.from_coords(
            x_coords=[],
            y_coords=[],
            values=[],
            timestamp='201501011200',
            mask_type='initiation',
        )
        assert sm.bbox == (0, 0, 0, 0)

    def test_from_coords_stores_fields(self):
        sm = SparseMask.from_coords(
            x_coords=[0],
            y_coords=[0],
            values=[99.0],
            timestamp='201501011200',
            mask_type='continuation',
        )
        assert sm.timestamp == '201501011200'
        assert sm.mask_type == 'continuation'
        assert list(sm.data_values) == [99.0]
