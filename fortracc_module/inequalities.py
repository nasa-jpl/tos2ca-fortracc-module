from typing import List, Optional
# from warnings import deprecated

import numpy as np
from numpy.core.multiarray import array as array

from fortracc_module.deprecation import deprecated
from fortracc_module.flow import TimeOrderedSequence
from fortracc_module.objects import GeoGrid


@deprecated("`LessThanEvent` has been deprecated in favor of `LessThanDetector`.")
class LessThanEvent(TimeOrderedSequence):
    """
    Specific TimeOrderedSequence object that defines phenomenon based on a threshold.  Any pixel that is
    less than the provided threshold is treated as a part of the phenomenon.
    """
    event_name = 'less_than_threshold'

    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float,
    ):
        """
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        masks = [image < threshold for image in images]
        super().__init__(masks, timestamps, grid)


@deprecated("`GreaterThanEvent` has been deprecated in favor of `GreaterThanDetector`.")
class GreaterThanEvent(TimeOrderedSequence):
    """
    Specific TimeOrderedSequence object that defines phenomenon based on a threshold.  Any pixel that is
    greater than the provided threshold is treated as a part of the phenomenon.
    """
    event_name = 'greater_than_threshold'

    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        masks = [image > threshold for image in images]
        super().__init__(masks, timestamps, grid)


@deprecated("`AnomalyEvent` has been deprecated in favor of `AnomalyDetector`.")
class AnomalyEvent(TimeOrderedSequence):
    """
    Specific TimeOrderedSequence object that defines phenomenon based on a threshold that is calculated per pixel
    as the standard deviation.  The threshold is compared to the images minus the pixel wise mean.  Any pixel that is
    less than the threshold is treated as a part of the phenomenon.
    """
    event_name = 'anomaly_event'

    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            frac_std: Optional[float] = 2.0
    ):
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)

        masks = [(image - mu) < (frac_std * std) for image in images]
        super().__init__(masks, timestamps, grid)
