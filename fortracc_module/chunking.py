from typing import List, Tuple
from warnings import warn

from fortracc_module.flow import SparseTimeOrderedSequence


def create_file_chunks(
    fns: List[Tuple[str, str]],
    num_per_chunk: int = 20,
) -> List[List[Tuple[str, str]]]:
    """
    Given a list of files, returns groups of files that may be treated as independant ForTraCC jobs
    then later stitched back together into a single `SparseTimeOrderedSequence` object.

    Parameters
    ----------
    fns: List[Tuple[str, str]]
        A list of tuples where the first element is the timestamp and the second element is the filename
        containing data at that timestamp.
    num_per_chunk: int
        The maximum number of files to include per job/chunk.
    
    Returns
    -------
        A list of lists representing the grouped files.
    """
    srt_fns = sorted(fns, key=lambda x: x[0])

    num_fns = len(srt_fns)
    if num_fns <= num_per_chunk:
        return srt_fns
    
    num_chunks = num_fns // num_per_chunk
    num_chunks += 1 if (num_fns % num_per_chunk) > 0 else 0

    eff_num_masks = (2 * (num_chunks - 1)) + num_fns
    eff_num_chunks = eff_num_masks // num_per_chunk
    eff_num_chunks += 1 if eff_num_masks % num_per_chunk > 0 else 0

    chunked_fn_list = []
    for i in range(eff_num_chunks):
        start_idx = i * (num_per_chunk - 1)
        end_idx = start_idx + num_per_chunk

        chunked_fn_list.append(
            [
                x for x in srt_fns[start_idx:end_idx]
            ]
        )
    return chunked_fn_list


def stitch(
    tos_list: List[SparseTimeOrderedSequence]
) -> SparseTimeOrderedSequence:
    """
    Given an ordered list of `SparseTimeOrderedSequence` objects, stitch them together into
    a single `SparseTimeOrderedSequence` object.
    
    Parameters
    ----------
    tos_list: List[SparseTimeOrderedSequence]
        List of `SparseTimeOrderedSequence` objects to stitch together.
    
    Returns
    -------
        A single `SparseTimeOrderedSequence` object representing the entire time series.
    """
    # Verify that `tos_list` is properly grouped
    grid = tos_list[0].grid
    detector_type = tos_list[0].detector_type
    if not detector_type:
        warn(
            "May be stitching together chunks with a different detector type since " +
            "detector type not specified."
        )
    
    last_timestamp = tos_list[0].timestamps[0]
    for tos in tos_list:
        # Ensure the sequences are defined over the same grid
        if tos.grid != grid:
            raise ValueError(
                "All objects need to be defined on the same grid."
            )
        
        # Ensure the sequences are defined using the same detector
        if tos.detector_type != detector_type:
            raise ValueError(
                "All objects need to have the same `detector_type`."
            )

        # Ensure the sequences are overlapping by exactly one time
        if tos.timestamps[0] != last_timestamp:
            raise ValueError(
                "Objects need to be overlapping, with exactly one mask shared " +
                "between successive sequences."
            )
        last_timestamp = tos.timestamps[-1]
    
    # Stitch together time series
    events = []
    timestamps = [
        tos_list[0].timestamps[0]
    ]
    time_series_map = dict()

    def _get_event_key(event):
        return (event.timestamp, *event.bbox)

    for tos in tos_list:
        last_timestamp = tos.timestamps[-1]

        for tos_event in tos.events:
            event_key = _get_event_key(tos_event[0])
            
            if event_key not in time_series_map:
                # No need to stitch this to a previous event
                events.append(
                    tos_event
                )
                idx = len(events) - 1
            else:
                idx = time_series_map.pop(event_key)
                events[idx].extend(
                    tos_event[1:]  # [1:] so as not to include the overlapping event again
                )
            
            last_event_key = _get_event_key(tos_event[-1])
            if last_event_key[0] == last_timestamp:
                time_series_map[last_event_key] = idx
        timestamps.extend(
            tos.timestamps[1:]
        )
    
    return SparseTimeOrderedSequence(
        events,
        timestamps,
        grid,
        detector_type=detector_type
    )
