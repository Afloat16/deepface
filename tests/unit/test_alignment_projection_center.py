"""Keep projected face crops aligned with the OpenCV image rotation."""

# third-party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface.modules import detection


@pytest.mark.parametrize("size", [(17, 13), (18, 13), (17, 14), (18, 14)])
@pytest.mark.parametrize("angle", [37.0, -29.5, 126.0, -151.0, 0.0])
def test_projection_uses_image_rotation_center(size, angle):
    height, width = size
    region = (4, 5, 11, 14)
    center = np.array([[[(region[0] + region[2]) / 2, (region[1] + region[3]) / 2]]])
    rotation = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    projected = cv2.transform(center, rotation)[0, 0]
    expected = (
        max(int(projected[0] - 3.5), 0),
        max(int(projected[1] - 4.5), 0),
        min(int(projected[0] + 3.5), width),
        min(int(projected[1] + 4.5), height),
    )
    assert detection.project_facial_area(region, angle, size) == expected
