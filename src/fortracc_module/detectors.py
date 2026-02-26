from abc import ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np


class Detector(metaclass=ABCMeta):
    """
    Base class for different detection methods.
    """
    name = ""

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def create_masks(
        self,
        images: List[np.array],
    ) -> List[np.array]:
        """
        Applies some prescription to generate 0/1 masks from images representing real 
        data(e.g. brightness temperature).

        Parameters
        ----------
        images: List[np.array]
            A list of images given as np.arrays which represent some data(e.g. brightness temperatures).
        
        Returns
        -------
            A list of masks given as np.arrays.
        """
        pass
    

class LessThanDetector(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    less than the provided threshold is treated as a part of the phenomenon.
    """
    name = "less_than_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold below which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image < self.threshold for image in images]


class GreaterThanDetector(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    greater than the provided threshold is treated as a part of the phenomenon.
    """
    name = "greater_than_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold above which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image > self.threshold for image in images]


class AnomalyDetector(Detector):
    """
    `Detector` which defines phenomenon based on a threshold that is calculated per pixel
    as the standard deviation.  The threshold is compared to the images minus the pixel wise mean.  
    Any pixel that is less than the threshold is treated as a part of the phenomenon.
    """
    name = "anomaly"

    def __init__(
        self,
        frac_std: Optional[float] = 2.0,
    ):
        """
        Parameters
        ----------
        frac_std: Optional[float]
            Used as a multiplier to the pixel-wise standard deviation which defines the threshold
            below which a phenomenon is defined.
        """
        self.frac_std = frac_std
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)

        return [(image - mu) < (self.frac_std * std) for image in images]
