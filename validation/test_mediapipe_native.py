"""Real MediaPipe graphs at a deliberately overlapping request boundary."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

import deepface
from deepface.models.face_detection.external.MediaPipe import MediaPipeClient


@pytest.mark.parametrize("selection", [0, 1])
def test_native_graph_preserves_concurrent_request_timestamps(monkeypatch, selection):
    monkeypatch.setenv("MEDIAPIPE_MODEL_SELECTION", str(selection))
    client = MediaPipeClient()
    image = cv2.imread(str(Path(deepface.__file__).parent.parent / "tests/unit/dataset/img1.jpg"))
    assert image is not None
    expected = client.detect_faces(image)
    assert expected
    first_entered, second_entered, second_started, release_first = (Event() for _ in range(4))
    native_make_packet = client.model._make_packet

    def delayed_packet(*args, **kwargs):
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        return native_make_packet(*args, **kwargs)

    monkeypatch.setattr(client.model, "_make_packet", delayed_packet)

    def second_request():
        second_started.set()
        return client.detect_faces(np.zeros_like(image))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.detect_faces, image)
            try:
                assert first_entered.wait(5)
                second = pool.submit(second_request)
                assert second_started.wait(5)
                second_entered.wait(0.3)
            finally:
                release_first.set()
            assert first.result(timeout=5) == expected
            assert second.result(timeout=5) == []
    finally:
        try:
            client.model.close()
        except ValueError:
            # The unfixed native graph is poisoned by a duplicate timestamp.
            pass
