"""Real OpenCV detection, SFace ONNX embeddings, and on-disk gallery caches."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepface.commons import folder_utils
from deepface.modules import recognition


@pytest.mark.parametrize("first_enforced", [False, True])
def test_native_gallery_policy(tmp_path, first_enforced):
    folder_utils.initialize_folder()
    source = cv2.imread("tests/unit/dataset/img1.jpg")
    assert source is not None
    image = cv2.resize(source, None, fx=0.25, fy=0.25)
    query = str(tmp_path / "face.jpg")
    assert cv2.imwrite(query, image)
    assert cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((128, 128, 3), np.uint8))

    for enforced in (first_enforced, not first_enforced):
        rows = recognition.find(
            query,
            str(tmp_path),
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=enforced,
            align=False,
            distance_metric="euclidean",
            return_type="dict",
            similarity_search=True,
            silent=True,
        )[0]
        distances = {Path(row["identity"]).name: row["distance"] for row in rows}
        print({"enforced": enforced, "distances": distances})
        assert distances["face.jpg"] == pytest.approx(0.0, abs=1e-6)
        assert math.isfinite(distances["blank.png"]) is (not enforced)
