"""Anti-spoofing crops keep the pixel scale expected by all three backends."""

import cv2
import numpy as np
import pytest

from deepface.models.spoofing.FasNetUtils import crop


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [2.7, 4.0])
@pytest.mark.parametrize("bbox", [(12, 15, 20, 25), (0, 0, 24, 28)])
def test_normalized_float_crops_match_raw_float_pixels(dtype, scale, bbox):
    image = np.random.default_rng(42).integers(0, 256, (60, 80, 3)).astype(dtype)
    normalized = image / 255
    original = normalized.copy()

    expected = crop(image, bbox, scale, 80, 80)
    actual = crop(normalized, bbox, scale, 80, 80)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=3e-5)
    np.testing.assert_array_equal(normalized, original)


@pytest.mark.parametrize("value", [0, 1, 128, 255])
def test_byte_pixels_are_not_rescaled(value):
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    actual = crop(image, (4, 4, 20, 20), 2.7, 80, 80)
    np.testing.assert_array_equal(actual, np.full((80, 80, 3), value, dtype=np.uint8))


def test_raw_float_image_uses_full_image_scale_before_cropping():
    image = np.full((200, 200, 3), 0.5, dtype=np.float32)
    image[0, 0] = 255
    actual = crop(image, (80, 80, 20, 20), 2.7, 80, 80)
    np.testing.assert_array_equal(actual, np.full((80, 80, 3), 0.5, dtype=np.float32))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_normalized_float_endpoints(value):
    image = np.full((32, 32, 3), value, dtype=np.float32)
    actual = crop(image, (4, 4, 20, 20), 2.7, 80, 80)
    np.testing.assert_array_equal(actual, np.full((80, 80, 3), value * 255))


def test_negative_float_input_is_not_mistaken_for_normalized_pixels():
    image = np.full((32, 32, 3), -0.5, dtype=np.float32)
    actual = crop(image, (4, 4, 20, 20), 2.7, 80, 80)
    np.testing.assert_array_equal(actual, np.full((80, 80, 3), -0.5))


def test_normalized_crop_keeps_channel_order():
    image = np.empty((32, 32, 3), dtype=np.float32)
    image[:] = [0.0, 0.5, 1.0]
    actual = crop(image, (4, 4, 20, 20), 2.7, 80, 80)
    expected = np.empty((80, 80, 3), dtype=np.float32)
    expected[:] = [0.0, 127.5, 255.0]
    np.testing.assert_array_equal(actual, expected)


def test_byte_resize_still_uses_opencv_integer_interpolation():
    image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    # With a whole-image box, the crop helper includes both original boundaries.
    actual = crop(image, (0, 0, 32, 32), 2.7, 80, 80)
    np.testing.assert_array_equal(actual, cv2.resize(image, (80, 80)))
