import numpy as np
import pytest

from fortracc_module.detectors import LessThanDetector, GreaterThanDetector, AnomalyDetector


def make_images():
    img1 = np.array([[1.0, 5.0], [10.0, 20.0]])
    img2 = np.array([[3.0, 7.0], [15.0, 25.0]])
    return [img1, img2]


class TestLessThanDetector:
    def test_name(self):
        assert LessThanDetector(10.0).name == "less_than_threshold"

    def test_basic_mask(self):
        images = make_images()
        det = LessThanDetector(threshold=8.0)
        masks = det.create_masks(images)
        assert masks[0].tolist() == [[True, True], [False, False]]
        assert masks[1].tolist() == [[True, True], [False, False]]

    def test_returns_same_count(self):
        images = make_images()
        masks = LessThanDetector(50.0).create_masks(images)
        assert len(masks) == len(images)

    def test_all_false_when_threshold_below_min(self):
        images = [np.array([[5.0, 5.0]])]
        masks = LessThanDetector(threshold=0.0).create_masks(images)
        assert not masks[0].any()

    def test_all_true_when_threshold_above_max(self):
        images = [np.array([[5.0, 5.0]])]
        masks = LessThanDetector(threshold=100.0).create_masks(images)
        assert masks[0].all()


class TestGreaterThanDetector:
    def test_name(self):
        assert GreaterThanDetector(10.0).name == "greater_than_threshold"

    def test_basic_mask(self):
        images = make_images()
        det = GreaterThanDetector(threshold=8.0)
        masks = det.create_masks(images)
        assert masks[0].tolist() == [[False, False], [True, True]]
        assert masks[1].tolist() == [[False, False], [True, True]]

    def test_complementary_to_less_than(self):
        images = [np.array([[1.0, 5.0, 9.0]])]
        threshold = 5.0
        lt_masks = LessThanDetector(threshold).create_masks(images)
        gt_masks = GreaterThanDetector(threshold).create_masks(images)
        combined = lt_masks[0] | gt_masks[0]
        # pixels equal to threshold are in neither
        assert combined.tolist() == [[True, False, True]]


class TestAnomalyDetector:
    def test_name(self):
        assert AnomalyDetector().name == "anomaly"

    def test_output_shape(self):
        images = make_images()
        masks = AnomalyDetector(frac_std=1.0).create_masks(images)
        assert len(masks) == len(images)
        assert masks[0].shape == images[0].shape

    def test_constant_image_all_true(self):
        # std == 0 everywhere, so (image - mu) == 0 < 0 * 0 == 0 is False for all
        # but with frac_std=2, 0 < 0 is False for each pixel
        images = [np.ones((3, 3)) * 5.0, np.ones((3, 3)) * 5.0]
        masks = AnomalyDetector(frac_std=2.0).create_masks(images)
        # (5 - 5) < (2 * 0) => 0 < 0 => False for all pixels
        assert not masks[0].any()
        assert not masks[1].any()

    def test_default_frac_std(self):
        det = AnomalyDetector()
        assert det.frac_std == 2.0
