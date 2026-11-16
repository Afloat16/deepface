"""Exercise real MediaPipe wheels without replacing their Python or native APIs."""

import json
import os
import sys

import cv2
import mediapipe
import numpy as np

from deepface.models.face_detection.external.MediaPipe import MediaPipeClient


def main():
    if sys.argv[1] == "incompatible":
        assert not hasattr(mediapipe, "solutions")
        try:
            MediaPipeClient()
        except ImportError as exc:
            assert "mediapipe>=0.8.7.3,<0.10.30" in str(exc)
            print(json.dumps({"sdk": mediapipe.__version__, "error": str(exc)}))
        else:
            raise AssertionError("Expected a compatible installation instruction")
        return

    image = cv2.imread("tests/unit/dataset/img1.jpg")
    assert image is not None
    for selection in (0, 1):
        os.environ["MEDIAPIPE_MODEL_SELECTION"] = str(selection)
        client = MediaPipeClient()
        try:
            faces = client.detect_faces(image)
            assert len(faces) == 1
            assert np.isfinite(faces[0].confidence) and faces[0].confidence > 0.7
            assert client.detect_faces(np.zeros_like(image)) == []
            print(
                json.dumps(
                    {
                        "sdk": mediapipe.__version__,
                        "selection": selection,
                        "faces": len(faces),
                        "confidence": faces[0].confidence,
                        "blank_faces": 0,
                    }
                )
            )
        finally:
            client.model.close()


if __name__ == "__main__":
    main()
