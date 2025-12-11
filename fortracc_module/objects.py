from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from skimage import measure


class GeoGrid:
    """
    Object used to store geographical lat/lon coordinates.  For now it's a glorified dictionary.
    """
    def __init__(
        self,
        latitude: List,
        longitude: List
    ):
        """
        :param List latitude: The latitude for each pixel on a grid given as a vector.
        :param List longitude: The longitude for each pixel on a grid given as a vector.
        """
        self._latitude = latitude
        self._longitude = longitude
        self._lat_bounds = (min(latitude), max(latitude))
        self._lon_bounds = (min(longitude), max(longitude))

        self._shape = (len(self._latitude), len(self._longitude))
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, GeoGrid):
            eq_flag = self._lat_bounds == __value._lat_bounds
            eq_flag &= self._lon_bounds == __value._lon_bounds
            eq_flag &= self._shape == __value._shape
            return eq_flag
        raise ValueError("Can only compare two `GeoGrid` objects.")
    
    @property
    def latitude(self):
        return self._latitude
    
    @property
    def longitude(self):
        return self._longitude
    
    @property
    def lat_bounds(self):
        return self._lat_bounds
    
    @property
    def lon_bounds(self):
        return self._lon_bounds
    
    @property
    def shape(self):
        return self._shape


class SparseGeoGrid(GeoGrid):
    @classmethod
    def from_grid(
        cls,
        grid: GeoGrid,
    ):
        return cls(
            grid.lat_bounds,
            grid.lon_bounds,
            grid.shape
        )
    
    @classmethod
    def from_lat_lon(
        cls,
        latitude: List,
        longitude: List,
    ):
        lat_bounds = (min(latitude), max(latitude))
        lon_bounds = (min(longitude), max(longitude))
        shape = (len(latitude), len(longitude))
        
        return cls(
            lat_bounds,
            lon_bounds,
            shape
        )

    def __init__(
        self,
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        shape: Tuple[int, int],
    ):
        self._lat_bounds = lat_bounds
        self._lon_bounds = lon_bounds
        self._shape = shape
    
    @property
    def latitude(self):
        return list(
            np.linspace(
                self._lat_bounds[0],
                self._lat_bounds[1],
                num=self._shape[0]
            )
        )
    
    @property
    def longitude(self):
        return list(
            np.linspace(
                self._lon_bounds[0],
                self._lon_bounds[1],
                num=self._shape[1]
            )
        )


@dataclass(frozen=True)
class SparseMask:
    row_inds: List[int]
    col_inds: List[int]
    data_values: List[Any]
    timestamp: str
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    mask_type: str  # intiation, continuation, merge, etc.

    @classmethod
    def from_regionprops(
        cls,
        props,
        data_image: np.array,
        timestamp: str,
        mask_type: str,
    ):
        nrows, ncols = data_image.shape

        bbox = props.bbox
        min_row, min_col, max_row, max_col = bbox

        mask = props.image
        full_mask = np.full((nrows, ncols), False)
        full_mask[min_row:max_row, min_col:max_col][mask] = True

        col_inds, row_inds = np.meshgrid(
            np.arange(ncols),
            np.arange(nrows)
        )
        row_inds = row_inds[full_mask]
        col_inds = col_inds[full_mask]
        data_values = data_image[full_mask]

        return cls(
            row_inds,
            col_inds,
            data_values,
            timestamp,
            bbox,
            mask_type,
        )
    @classmethod
    def from_coords(
        cls,
        x_coords: List[int],
        y_coords: List[int],
        values: List[float],
        timestamp: str,
        mask_type: str,
        properties: Optional[dict] = None,
    ):
        if len(x_coords) == 0 or len(y_coords) == 0:
            bbox = (0, 0, 0, 0)
        else:
            min_row = min(y_coords)
            max_row = max(y_coords)
            min_col = min(x_coords)
            max_col = max(x_coords)
            bbox = (min_row, min_col, max_row + 1, max_col + 1)  # assuming exclusive max

        return cls(
            row_inds=y_coords,
            col_inds=x_coords,
            data_values=values,
            timestamp=timestamp,
            bbox=bbox,
            mask_type=mask_type
        )


class Scene:
    """
    Object used for describing a phenomenon defined by a mask.  Stores individual "events"
    as connected components of the provided mask and filters out events below a certain size
    (total number of pixels in connected component).
    """
    def __init__(
        self,
        mask: np.array,
        timestamp: Optional[str] = None,
        connectivity: Optional[int] = 2,
        min_size: Optional[int] = 1
    ):
        """
        :param np.array mask: True/False mask describing a phenomenon.
        :param str timestamp: A string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                              provided mask.
        :param int connectivity: Determines how nearest neighbors are chosen when building connected components.
                                 Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
                                 the diagonals as well.
        :param int min_size: Smallest size connected component(total number of pixels) to include as an "event".
        """
        self.mask = mask
        self.timestamp = timestamp
        self.connectivity = connectivity
        self.min_size = min_size

        self.labels, self.events = self._connected_components()

    def __getitem__(self, item):
        return self.events[item]

    def _connected_components(self):
        """
        Uses scikit-image to calculate the connected components of the provided mask.

        :return np.array labels: An np.array the same size as mask where each pixel is a unique string
                                 that determines which "event"(i.e. connected component) the pixel belongs to.
                                 The string is formatted as {timestamp}.{id} where `timestamp` is the provided
                                 timestamp and `id` is the number between 1 and the total number of events.
                                 If the pixel is a part of the background, the string is 'None'.
        :return dict events: A dictionary that maps an event id to the event itself.  The event id is formatted
                             as {timestamp}.{id} where `timestamp` is the provided timestamp and `id` is the
                             number between 1 and the total number of events.  The event is a RegionProperties
                             object that contains many attributes for the connected component.  See
                             https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
                             for a full list of properties.
        """
        labels, num_events = measure.label(
            self.mask,
            return_num=True,
            connectivity=self.connectivity
        )
        events = measure.regionprops(labels)
        assert len(events) == num_events

        # Assign unique ID to each component
        event_ids = ['None']
        event_ids.extend(
            [
                f'{self.timestamp}.{i + 1}' if events[i].area >= self.min_size else 'None'
                for i in range(num_events)
            ]
        )
        labels = np.array(event_ids)[labels]
        events = {
            c_id: x for c_id, x in zip(event_ids[1:], events)  # Skip the first 'None' event
            if c_id != 'None'
        }

        return labels, events
