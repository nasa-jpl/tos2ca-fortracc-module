from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from fortracc_module.detectors import Detector
from fortracc_module.objects import GeoGrid, Scene


class FortraccRunner:
    """
    Runs ForTraCC and saves the inputs and outputs in the class variables.
    """

    def __init__(
        self,
        images: List[np.array],
        timestamps: List[str],
        grid: GeoGrid,
        detector: Detector,
        connectivity: Optional[int] = 2,
        min_olap: Optional[float] = 0.25,
        min_size: Optional[int] = 150,
    ):
        """
        Parameters
        ----------
        images: List[np.array]
            A list of images given as np.arrays which represent some data(e.g. brightness temperatures).
        timestamps: List[str]
            A list of timestamps with the same size as `masks` where each item is a
            string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
            corresponding mask.
        grid: GeoGrid
            A `GeoGrid` object that defines the geographical grid on which the images live.
        detector: Detector
            The `Detector` to use when identifying phenomenon in the images.
        connectivity: Optional[int]
            Determines how nearest neighbors are chosen when building connected components.
            Choosing 1 selects top/bottom, left/right neighbors whereas 2 includes
            the diagonals as well.
        min_olap: Optional[float]
            The smallest amount of overlap(number of overlapping pixels relative to the size
            of the first event) to consider as valid.  Any overlap below this value is ignored.
        min_size: Optional[int]
            Smallest size connected component(total number of pixels) to include as an "event".
        """
        # Run checks
        if len(images) != len(timestamps):
            raise ValueError(
                "The number of images must be equal to the number of timestamps." +
                f" Got {len(images)} images and {len(timestamps)} timestamps."
            )
        for image in images:
            if image.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for masks.  Got {image.ndim}D array instead."
                )
        for t in timestamps:
            if len(t) != 12:
                raise ValueError(
                    "Expected all timestamps to be formatted as YYYYMMDDhhmm(e.g. 201501011430)."
                )
        
        # Initialize input atributes
        self.images = images
        self.timestamps = timestamps
        self.grid = grid
        self.detector = detector
        self.connectivity = connectivity
        self.min_olap = min_olap
        self.min_size = min_size

        # Initialize calculated attributes
        self.masks = detector.create_masks(images)
        self.time_series = []

        # Initialize helper attributes
        self._time_series_map = dict()  # Last added event_id in series -> time_series index

        # Run ForTraCC
        self._run()

    def _build_scenes(
        self,
    ) -> List[Scene]:
        """
        Converts the provided masks into `Scene` objects used to define events.  If need be, this can
        be modified to run in parallel for a large amount of masks.

        Returns
        -------
            A list of `Scene` objects used to define an event.
        """
        # Can be processed in parallel if need be
        return [
            Scene(
                mask,
                timestamp=self.timestamps[i],
                connectivity=self.connectivity,
                min_size=self.min_size
            )
            for i, mask in enumerate(self.masks)
        ]

    def _get_overlap_map(
        self,
        prev_scene: Scene,
        scene: Scene,
    ) -> Dict:
        """
        Calculates the overlap between events in any two consecutive scenes and creates a mapping
        between any event in one scene to the events it overlaps with in the other.

        Parameters
        ----------
        prev_scene: Scene
            The first `Scene` object in the provided consecutive pair.
        scene: Scene
            The second `Scene` object in the provided consecutive pair.

        Returns
        -------
            A dictionary that maps any event in one scene to the events it overlaps with in the other.
        """
        overlap = prev_scene.mask & scene.mask
        olap_scene = Scene(
            overlap,
            connectivity=self.connectivity
        )

        olap_map = dict()  # event_id -> Dict[event_ids]
        for olap in olap_scene.events.values():
            min_row, min_col, max_row, max_col = olap.bbox

            prev_event_ids = set(
                prev_scene.labels[min_row:max_row, min_col:max_col][olap.image]
            )
            crnt_event_ids = set(
                scene.labels[min_row:max_row, min_col:max_col][olap.image]
            )
            assert len(prev_event_ids) == len(crnt_event_ids) == 1

            prev_event_id = list(prev_event_ids)[0]
            if prev_event_id == 'None':
                continue

            crnt_event_id = list(crnt_event_ids)[0]
            if crnt_event_id == 'None':
                continue

            # Forward mapping for "split" and "dissipation" events
            if prev_event_id not in olap_map:
                olap_map[prev_event_id] = dict()
            if crnt_event_id in olap_map[prev_event_id]:
                olap_map[prev_event_id][crnt_event_id]['olap_area'] += olap.area
                olap_map[prev_event_id][crnt_event_id]['prev_area'] += prev_scene[prev_event_id].area
            else:
                olap_map[prev_event_id][crnt_event_id] = {
                    'event_id': crnt_event_id,
                    'event': scene[crnt_event_id],
                    'olap_area': olap.area,
                    'prev_area': prev_scene[prev_event_id].area
                }

            # Backward mapping for "merge" and "initiation" events
            if crnt_event_id not in olap_map:
                olap_map[crnt_event_id] = dict()
            if prev_event_id in olap_map[crnt_event_id]:
                olap_map[crnt_event_id][prev_event_id]['olap_area'] += olap.area
                olap_map[crnt_event_id][prev_event_id]['prev_area'] += olap.area
            else:
                olap_map[crnt_event_id][prev_event_id] = {
                    'event_id': prev_event_id,
                    'event': prev_scene[prev_event_id],
                    'olap_area': olap.area,
                    'prev_area': prev_scene[prev_event_id].area
                }

        # Filtering based on final fractional overlap
        olap_map_keys = list(olap_map.keys())
        for event_id in olap_map_keys:
            olap_event_keys = list(olap_map[event_id].keys())
            for k in olap_event_keys:
                v = olap_map[event_id][k]
                frac_olap = v.pop('olap_area') / v.pop('prev_area')
                if frac_olap < self.min_olap:
                    olap_map[event_id].pop(k)
                    continue
                v['overlap'] = frac_olap
            if len(olap_map[event_id]) == 0:
                olap_map.pop(event_id)
        return olap_map

    def _add_initiation(
        self,
        event_id: str,
        event,
        tag: str,
    ) -> Set[str]:
        """
        Adds an "initiation" event to the time series.

        Parameters
        ----------
        event_id: str
            The event ID of the initiated event.
        event: RegionProperties
            The event to add to the time series.  The event is a RegionProperties
            object that contains many attributes for the connected component.  See
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            for a full list of properties.
        tag: str
            The name to use for describing the event.

        Returns
        -------
            A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert event_id not in self._time_series_map

        self.time_series.append(
            [(event_id, event, tag)]
        )
        self._time_series_map[event_id] = len(self.time_series) - 1
        return {event_id}

    def _add_continuation(
        self, 
        prev_event_id: str,
        event_id: str,
        event,
        tag: str,
    ) -> Set[str]:
        """
        Adds a "continuation" event to the time series.

        Paramters
        ---------
        prev_event_id: str
            The event ID of the continuing event used to identify the time series.
        event_id: str
            The event ID of the new event that is being added as a continuation of the time series.
        event: RegionProperties
            The event to add to the time series.  The event is a RegionProperties
            object that contains many attributes for the connected component.  See
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            for a full list of properties.
        tag: str
            The name to use for describing the event.

        Returns
        -------
            A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert prev_event_id in self._time_series_map

        time_series_idx = self._time_series_map.pop(prev_event_id)
        self.time_series[time_series_idx].append(
            (event_id, event, tag)
        )
        self._time_series_map[event_id] = time_series_idx

        return {prev_event_id, event_id}

    def _add_dissipation(
        self,
        event_id: str,
        tag: str,
    ) -> Set[str]:
        """
        Adds a "dissipation" event to the time series.

        Parameters
        ----------
        event_id: str
            The event ID to label as "dissipation".
        tag: str
            The name to use for describing the event.

        Returns
        -------
            A set of events to remove from the pool of events which have yet to be labeled.
        """
        assert event_id in self._time_series_map

        time_series_idx = self._time_series_map[event_id]
        _, event, _ = self.time_series[time_series_idx][-1]
        self.time_series[time_series_idx][-1] = (
            event_id, event, tag
        )

        return {event_id}

    @staticmethod
    def _separate_max_olap(
        events: List[dict],
    ) -> Tuple[dict, List[dict]]:
        """
        Separates the event with the largest overlap from the others.

        Parameters
        ----------
        events: List[dict]
            A list of events each given as a dictionary with keys `event_id`, `event` and `overlap`.

        Returns
        -------
            Returns the event with the largest overlap and a list of the other events.
        """
        max_frac_olap = 0.0
        max_idx = 0
        for i, item in enumerate(events):
            if item['overlap'] > max_frac_olap:
                max_idx = i
                max_frac_olap = item['overlap']
        return events.pop(max_idx), events

    def _add_merge(
        self,
        mapped_events: List[dict],
        event_id: str,
        event,
    ) -> Set[str]:
        """
        Adds a "merge" event to the time series.

        Parameters
        ----------
        mapped_events: List[dict]
            A list of events each given as a dictionary with keys `event_id`, `event` and `overlap`.
        event_id: str
            The event ID which indicates the final merged event.
        event: RegionProperties
            The event to add to the time series.  The event is a RegionProperties
            object that contains many attributes for the connected component.  See
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            for a full list of properties.

        Returns
        -------
            A set of events to remove from the pool of events which have yet to be labeled.
        """
        max_olap_event, events = self._separate_max_olap(mapped_events)
        labeled_events = set()

        # The event with the maximum overlap gets continued
        labeled_events.update(
            self._add_continuation(
                max_olap_event['event_id'],
                event_id,
                event,
                'continuation - merge'
            )
        )

        # The other events dissipate in a merge
        for item in events:
            prev_event_id = item['event_id']
            labeled_events.update(
                self._add_dissipation(prev_event_id, 'dissipation - merge')
            )
        return labeled_events

    def _add_split(
        self,
        mapped_events: List[dict],
        event_id: str,
    ) -> Set[str]:
        """
        Adds a "split" event to the time series.

        Parameters
        ----------
        mapped_events: List[dict]
            A list of events each given as a dictionary with keys `event_id`, `event` and `overlap`.
        event_id: str
            The event ID which indicates the initial event that later splits into multiple events.

        Returns
        -------
            A set of events to remove from the pool of events which have yet to be labeled.
        """
        max_olap_event, events = self._separate_max_olap(mapped_events)
        labeled_events = set()

        # The event with the maximum overlap gets continued
        labeled_events.update(
            self._add_continuation(
                event_id,
                max_olap_event['event_id'],
                max_olap_event['event'],
                'continuation - split'
            )
        )

        # The other events initiate with a split
        for item in events:
            next_event_id = item['event_id']
            labeled_events.update(
                self._add_initiation(next_event_id, item['event'], 'initiation - split')
            )
        return labeled_events

    def _run(
        self,
    ) -> None:
        """
        Runs ForTraCC.
        """
        all_scenes = self._build_scenes()

        prev_scene = Scene(
            np.zeros(self.grid.shape, dtype=bool),
            connectivity=self.connectivity
        )
        for scene in all_scenes:
            # Create mapping between overlapping events
            olap_map = self._get_overlap_map(
                prev_scene,
                scene,
            )

            # Gather all the events between the two images and sort them in descending order by size
            all_events = list(
                prev_scene.events.items()
            )
            all_events.extend(
                list(scene.events.items())
            )
            all_events = OrderedDict(
                sorted(all_events, key=lambda x: x[1].area)
            )  # [smallest, ... , biggest]

            # Go through each event and add to existing time series
            crnt_timestamp = int(scene.timestamp)
            while len(all_events) > 0:
                event_id, event = all_events.popitem()

                event_timestamp = int(event_id.split('.')[0])
                is_new = event_timestamp == crnt_timestamp
                mapped_events = []
                if event_id in olap_map:
                    mapped_events.extend(
                        [
                            olaps for olap_event_id, olaps in olap_map[event_id].items()
                            if olap_event_id in all_events
                        ]
                    )

                # Categorize the event relative to the events it connects to
                labeled_events = set()
                if is_new:
                    if len(mapped_events) == 0:
                        # Initiation of new time series
                        self._add_initiation(
                            event_id,
                            event,
                            'initiation'
                        )
                    elif len(mapped_events) == 1:
                        # Continuation of previous event
                        prev_event_id = mapped_events[0]['event_id']
                        continue_events = self._add_continuation(
                            prev_event_id,
                            event_id,
                            event,
                            'continuation'
                        )
                        labeled_events.update(continue_events)
                    else:
                        # Merging of two or more events
                        merged_events = self._add_merge(
                            mapped_events,
                            event_id,
                            event
                        )
                        labeled_events.update(merged_events)
                else:
                    if len(mapped_events) == 0:
                        # Dissipation of time series
                        self._add_dissipation(event_id, 'dissipation')
                    elif len(mapped_events) == 1:
                        # Continuation of current event
                        next_event_id = mapped_events[0]['event_id']
                        continue_events = self._add_continuation(
                            event_id,
                            next_event_id,
                            mapped_events[0]['event'],
                            'continuation'
                        )
                        labeled_events.update(continue_events)
                    else:
                        # Splitting of two or more events
                        split_events = self._add_split(
                            mapped_events,
                            event_id
                        )
                        labeled_events.update(split_events)

                # Clean up the events that have been labeled this round
                for seen_event in labeled_events:
                    if seen_event in all_events:
                        all_events.pop(seen_event)
            prev_scene = scene
        return
