"""Concurrent CenterFace requests must retain their own image geometry."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from deepface import DeepFace
from deepface.models.face_detection.onnx.CenterFace import CenterFace, CenterFaceClient
from deepface.modules import modeling


class ControlledSession:
    """Return a centered face while allowing a second inference to finish first."""

    def __init__(self):
        self.block_first = False
        self.first_started = Event()
        self.release_first = Event()

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _outputs, inputs):
        blob = inputs["input"]
        if self.block_first and int(blob.max()) == 40:
            self.first_started.set()
            assert self.release_first.wait(10), "first inference was not released"

        height, width = blob.shape[2:]
        heatmap = np.zeros((1, 1, height // 4, width // 4), dtype=np.float32)
        heatmap[0, 0, height // 8, width // 8] = 0.9
        scale = np.full((1, 2, height // 4, width // 4), np.log(2), dtype=np.float32)
        offset = np.zeros_like(scale)
        landmarks = np.full((1, 10, height // 4, width // 4), 0.5, dtype=np.float32)
        landmarks[:, 1] = 0.25
        landmarks[:, 3] = 0.75
        return heatmap, scale, offset, landmarks


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize(
    "shapes",
    [((33, 65), (80, 50)), ((80, 50), (33, 65)), ((48, 48), (48, 48))],
)
def test_concurrent_extraction_matches_each_images_sequential_result(monkeypatch, align, shapes):
    session = ControlledSession()
    client = CenterFaceClient.__new__(CenterFaceClient)
    client.model = CenterFace(session=session)
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: client)
    images = [
        np.full((*shape, 3), value, dtype=np.uint8) for shape, value in zip(shapes, (40, 180))
    ]

    def extract(image):
        return DeepFace.extract_faces(
            img_path=image,
            detector_backend="centerface",
            align=align,
            color_face="bgr",
            normalize_face=False,
        )

    expected = [extract(image) for image in images]
    assert all(len(faces) == 1 for faces in expected)
    session.block_first = True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(extract, images[0])
        try:
            assert session.first_started.wait(10), "first inference did not start"
            # This request finishes while the first one is still inside inference.
            second = executor.submit(extract, images[1]).result(timeout=10)
        finally:
            session.release_first.set()
        actual = [first.result(timeout=10), second]

    for faces, baseline in zip(actual, expected):
        assert len(faces) == len(baseline)
        assert faces[0]["facial_area"] == baseline[0]["facial_area"]
        assert faces[0]["confidence"] == baseline[0]["confidence"]
        np.testing.assert_array_equal(faces[0]["face"], baseline[0]["face"])
