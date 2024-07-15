import sys
sys.path.insert(0, '/Users/sawaya/TOS2CA/fortracc-module')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from fortracc_module.chunking import create_file_chunks, stitch
from fortracc_module.detectors import LessThanDetector
from fortracc_module.flow import SparseTimeOrderedSequence
from fortracc_module.objects import SparseGeoGrid
from fortracc_module.utils import write_nc4


def mock_job_allocator():
    test_fns = sorted(
        [
            (fn[:-4], 'test_data/' + fn) 
            for fn in os.listdir('test_data') if fn[-4:] == '.txt'
        ]
    )
    assert len(test_fns) == 12

    return create_file_chunks(
        test_fns,
        num_per_chunk=3
    )


def mock_reader(fns):
    images = []
    for fn in fns:
        data = pd.read_csv(
            fn,
            sep='\t',
            header=None
        ).values

        images.append(data)
    return images


def mock_workers(job_inputs):
    results = []
    for inputs in job_inputs:
        s = time.time()
        results.append(
            SparseTimeOrderedSequence.run_fortracc(
                **inputs
            )
        )
        e = time.time()
        print(f'Elapsed time: {e - s:.4f}s')
    return results


def main():
    # Mock up the job inputs
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
    grid = SparseGeoGrid.from_lat_lon(
        lat,
        lon
    )
    detector = LessThanDetector(
        threshold=235
    )
    jobs = mock_job_allocator()

    # Create the inputs to each job
    print('Loading test data...')
    
    job_inputs = []
    for job in jobs:
        job_timestamps, job_files = list(zip(*job))
        job_images = mock_reader(job_files)

        job_inputs.append(
            {
                'images': job_images,
                'timestamps': job_timestamps,
                'grid': grid,
                'detector': detector,
                'connectivity': 2,
                'min_olap': 0.25,
                'min_size': 150,
            }
        )

    # Submit each job to a worker(here just a for-loop)
    results = mock_workers(job_inputs)

    # Stitch the disparate outputs into a single object
    stos = stitch(results)

    # Write the ForTraCC outputs to disk as a netCDF
    anomaly_table = write_nc4(
        stos,
        'fortracc_output.nc4',
        output_dir='chunked_outputs/',
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

        print(f'\tSaving image at {timestamp} in chunked_outputs/')
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gist_rainbow', norm=None, interpolation='none')
        plt.clim(1, len(stos.events))
        plt.savefig(f'chunked_outputs/{timestamp}.pdf')
        plt.tight_layout()
        plt.close()
    return


if __name__ == '__main__':
    main()
