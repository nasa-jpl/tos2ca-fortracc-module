import sys
sys.path.insert(0, '/Users/sawaya/TOS2CA/fortracc-module')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from fortracc_module.flow import SparseTimeOrderedSequence
from fortracc_module.detectors import LessThanDetector
from fortracc_module.objects import SparseGeoGrid
from fortracc_module.utils import write_nc4


def mock_reader():
    # Load in the test data
    print('Loading test data...')
    test_fns = sorted(
        [
            fn 
            for fn in os.listdir('test_data') if fn[-4:] == '.txt'
        ]
    )
    assert len(test_fns) == 12

    lat = list(
        np.unique(
            pd.read_csv(
                'test_data/navigation/lat.txt',
                header=None
            ).values.flatten()
        )
    )
    lon = list(
        np.unique(
            pd.read_csv(
                'test_data/navigation/lon.txt',
                header=None
            ).values.flatten()
        )
    )

    test_images = []
    t_stamps = []
    for fn in test_fns:
        data = pd.read_csv(
            'test_data/' + fn,
            sep='\t',
            header=None
        ).values
        assert data.shape == (len(lat), len(lon))

        test_images.append(data)
        t_stamps.append(
            fn[:-4]
        )
    return test_images, lat, lon, t_stamps


def main():
    # Load in data and intialize ForTraCC parameters
    images, lat, lon, timestamps = mock_reader()
    grid = SparseGeoGrid.from_lat_lon(
        lat,
        lon
    )
    detector = LessThanDetector(
        threshold=235
    )

    # Run ForTraCC
    print('Running ForTraCC...')
    s = time.time()
    stos = SparseTimeOrderedSequence.run_fortracc(
        images,
        timestamps,
        grid,
        detector,
        connectivity=2,
        min_olap=0.25,
        min_size=150,
        retain_run_params=False
    )
    e = time.time()
    print(f'Elapsed time: {e - s:.4f}s')

    # Write the ForTraCC outputs to disk as a netCDF
    anomaly_table = write_nc4(
        stos,
        'fortracc_output.nc4',
        output_dir='python_outputs/',
    )
    print("Anomaly Table")
    print('-' * 15)
    for entry in anomaly_table[:3]:
        print(entry)
        print()
    print()

    # Re-create the masks using the delineated events
    composite_mask = {
        timestamp: np.zeros(stos.grid.shape) for timestamp in stos.timestamps
    }
    for i, events in enumerate(stos.masks):
        for timestamp, mask in events.items():
            composite_mask[timestamp][mask] = i + 1

    # Plot the events as a series of PDFs
    print('Generating images...')
    for timestamp, image in composite_mask.items():
        image[image == 0] = None

        print(f'\tSaving image at {timestamp} in python_outputs/')
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gist_rainbow', norm=None, interpolation='none')
        plt.clim(1, len(stos.events))
        plt.savefig(f'python_outputs/{timestamp}.pdf')
        plt.tight_layout()
        plt.close()
    return


if __name__ == '__main__':
    main()
