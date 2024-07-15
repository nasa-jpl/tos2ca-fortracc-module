from typing import Optional, Union
from os.path import exists
from datetime import datetime

import netCDF4 as nc
import numpy as np
import os

from fortracc_module.deprecation import deprecated
from fortracc_module.flow import TimeOrderedSequence, SparseTimeOrderedSequence


@deprecated("`_mask2idx` will no longer be used in `write_nc4`.")
def _mask2idx(
    mask: np.array
):
    """
    Converts a 2D numpy array into a sparse matrix given as a set of indices and the value.

    :param np.array mask:  The input matrix given as a 2D numpy array.
    """
    assert mask.ndim == 2
    nrows, ncols = mask.shape

    indices = []
    for i in range(nrows):
        for j in range(ncols):
            if mask[i,j] > 0:
                indices.append(
                    (i, j, mask[i,j])
                )
    return np.array(indices, dtype=np.int32)


def write_nc4(
    tos: Union[TimeOrderedSequence, SparseTimeOrderedSequence],
    fn: str,
    output_dir: Optional[str] = '.',
    file_format: Optional[str] = 'NETCDF4',
    metadata: dict = {'jobID': None, 'variabile': None, 'dataset': None, 'threshold': None}
):
    """
    Utility for writing the ForTraCC outputs to a netCDF file.
    
    :param TimeOrderedSequence tos:  The ForTraCC object that contains the outputs to write to disk.
    :param str fn:  The output file name, excluding the path and including the file extension.
    :param Optional[str] output_dir:  The output path where the file will be stored.  Defaults to current
                                      working directory.
    :param Optional[str] file_format:  The netCDF file format to use.  See https://unidata.github.io/netcdf4-python
    :param dict metadata: Dictionary containing metadata elements
    """
    if isinstance(tos, TimeOrderedSequence):
        deprecated(
            "Passing in a `TimeOrderedSequence` to `write_nc4` is deprecated in favor of passing" + 
            " `SparseTimeOrderedSequence`."
        )
        if not tos.has_been_run:
            raise ValueError(
                """
                ForTraCC has not been run.  Please call `.run_fortracc` before passing the 
                TimeOrderedSequence object to write_to_nc4.
                """
            )
        if exists(f'{output_dir}/{fn}'):
            os.remove(f'{output_dir}/{fn}')
            # raise ValueError(
            #     f"""
            #     '{output_dir}/{fn}' already exists, will not overwrite.
            #     """
            # )
        lat = tos.grid.latitude
        lon = tos.grid.longitude
        timestamps = tos.timestamps

        all_masks = {timestamp: [] for timestamp in timestamps}
        anomaly_table = []
        for series_id, series in enumerate(tos.time_series):
            anomaly_table.append(
                {
                    'name': f'Anomaly {series_id + 1}',
                    'start_date': series[0][0].split('.')[0],
                    'end_date': series[-1][0].split('.')[0]
                }
            )

            for scene in series:
                event_id, event, _ = scene

                timestamp = event_id.split('.')[0]

                min_row, min_col, max_row, max_col = event.bbox

                mask = event.image
                tmp_image = np.zeros(tos.grid.shape)
                tmp_image[min_row:max_row, min_col:max_col][mask] = series_id + 1

                all_masks[timestamp].extend(
                    _mask2idx(tmp_image)
                )

        ncfile = nc.Dataset(
            f'{output_dir}/{fn}',
            mode='w',
            format=file_format
        )
        ncfile.event_type = tos.event_name
        ncfile.latitude_bounds = tos.grid.lat_bounds
        ncfile.longitude_bounds = tos.grid.lon_bounds
        ncfile.start_date = timestamps[0]
        ncfile.end_date = timestamps[-1]
        ncfile.format = 'netCDF-4'
        # ncfile.job_id = metadata['jobID']
        # ncfile.variable = metadata['variable']
        # ncfile.dataset = metadata['dataset']
        # ncfile.threshold = metadata['threshold']
        ncfile.website = 'https://tos2ca-dev1.jpl.nasa.gov'
        ncfile.project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
        ncfile.institution = 'NASA Jet Propulsion Laboratory'
        ncfile.production_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Navigation data
        nav_group = ncfile.createGroup('navigation')

        lat_dim = nav_group.createDimension('lat', len(lat))
        lat_var = nav_group.createVariable('lat', 'f4', ('lat',))
        lat_var[:] = np.array(lat, dtype=np.float32)

        lon_dim = nav_group.createDimension('lon', len(lon))
        lon_var = nav_group.createVariable('lon', 'f4', ('lon',))
        lon_var[:] = np.array(lon, dtype=np.float32)

        # Mask data
        mask_group = ncfile.createGroup('masks')
        mask_group.num_events = len(tos.time_series)
        for timestamp in timestamps:
            timestamp_mask_group = mask_group.createGroup(timestamp)
            indices = all_masks[timestamp]

            mask_ind_dim = timestamp_mask_group.createDimension('num_pixels', len(indices))
            mask_col_dim = timestamp_mask_group.createDimension('num_cols', 3)
            mask_var = timestamp_mask_group.createVariable('mask_indices', 'i4', ('num_pixels', 'num_cols',))
            mask_var.description = 'Each row is a pixel with the columns indicating (i, j, event_id).'
            mask_var[:] = indices
        ncfile.close()
        return anomaly_table
    else:
        fn = f'{output_dir}/{fn}'
        overwrite = True
        metadata = None

        if metadata:
            assert ('jobID' in metadata) and metadata['jobID']
            assert ('variable' in metadata) and metadata['variable']
            assert ('dataset' in metadata) and metadata['dataset']
            assert ('threshold' in metadata) and metadata['threshold']
        if os.path.exists(fn) and not overwrite:
            raise ValueError(
                f"`{fn}` already exists.  If you wish to overwrite, set `overwrite=True`."
            )
        
        # Form tuple to pass to NC file
        anomaly_table = []
        all_masks = {
            timestamp: [] for timestamp in tos.timestamps
        }
        for i, events in enumerate(tos.events):
            anomaly_table.append(
                {
                    'name': f'Anomaly {i + 1}',
                    'start_date': events[0].timestamp,
                    'end_date': events[-1].timestamp
                }
            )

            for mask in events:
                timestamp = mask.timestamp

                all_masks[timestamp] = [
                    (row_idx, col_idx, i + 1, data_val)
                    for row_idx, col_idx, data_val in zip(mask.row_inds, mask.col_inds, mask.data_values)
                ]
        
        # Build NC file
        ncfile = nc.Dataset(
            fn,
            mode='w',
            format='NETCDF4'
        )

        ## Global properties
        ncfile.event_type = tos.detector_type
        ncfile.latitude_bounds = tos.grid.lat_bounds
        ncfile.longitude_bounds = tos.grid.lon_bounds
        ncfile.start_date = tos.timestamps[0]
        ncfile.end_date = tos.timestamps[-1]
        ncfile.format = 'netCDF-4'
        ncfile.website = 'https://tos2ca-dev1.jpl.nasa.gov'
        ncfile.project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
        ncfile.institution = 'NASA Jet Propulsion Laboratory'
        ncfile.production_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if metadata:
            ncfile.job_id = metadata['jobID']
            ncfile.variable = metadata['variable']
            ncfile.dataset = metadata['dataset']
            ncfile.threshold = metadata['threshold']
        
        ## Lat/Lon
        nav_group = ncfile.createGroup('navigation')

        lat = tos.grid.latitude
        lat_dim = nav_group.createDimension('lat', len(lat))
        lat_var = nav_group.createVariable('lat', 'f4', ('lat',))
        lat_var[:] = np.array(lat, dtype=np.float32)

        lon = tos.grid.longitude
        lon_dim = nav_group.createDimension('lon', len(lon))
        lon_var = nav_group.createVariable('lon', 'f4', ('lon',))
        lon_var[:] = np.array(lon, dtype=np.float32)

        ## Mask data
        mask_group = ncfile.createGroup('masks')
        mask_group.num_events = len(tos.events)
        for timestamp in tos.timestamps:
            timestamp_mask_group = mask_group.createGroup(timestamp)
            indices = all_masks[timestamp]

            ### Integer fields
            mask_ind_dim = timestamp_mask_group.createDimension('num_pixels', len(indices))
            mask_col_dim = timestamp_mask_group.createDimension('num_cols', 3)
            mask_var = timestamp_mask_group.createVariable('mask_indices', 'i4', ('num_pixels', 'num_cols',))
            mask_var.description = 'Each row is a pixel with the columns indicating (i, j, event_id).'
            mask_var[:] = [(x[0], x[1], x[2]) for x in indices]

            ### Floats
            data_val_var = timestamp_mask_group.createVariable('data_values', 'f4', ('num_pixels',))
            data_val_var.description = "Each entry gives the corresponding data value of the mask pixel in `mask_indices`."
            data_val_var[:] = [x[-1] for x in indices]
        ncfile.close()
        return anomaly_table


def read_nc4(
    fn: str
):
    """
    Reads a netCDF file that has been saved to disk using `write_nc4`.

    :param str fn:  The name of the input file including both the path and the file extension.
    """
    ds = nc.Dataset(fn, mode='r')

    payload = {
        'lat': list(ds['navigation']['lat'][:]),
        'lon': list(ds['navigation']['lon'][:]),
        'total_num_events': ds['masks'].num_events
    }
    grid_shape = (len(payload['lat']), len(payload['lon']))

    send_depr_warning = False
    all_masks = dict()
    all_data_vals = dict()
    for timestamp, dat in ds['masks'].groups.items():
        all_data_vals[timestamp] = np.zeros(grid_shape)
        all_masks[timestamp] = np.zeros(grid_shape)
        if 'data_values' not in dat:
            send_depr_warning = True

            for i, j, event_id in dat['mask_indices'][:]:
                all_masks[timestamp][i, j] = event_id
        else:
            for (i, j, event_id), data_val in zip(dat['mask_indices'][:], dat['data_values'][:]):
                all_masks[timestamp][i, j] = event_id
                all_data_vals[timestamp][i, j] = data_val
    if send_depr_warning:
        deprecated(
            "`read_nc4` will only work when loading a file created with a " +
            "`SparseTimeOrderedSeqeunce` object in the future."
        )
    payload['masks'] = all_masks
    payload['data_values'] = all_data_vals
    return payload
