"""Compare actual concurrent Haar face/eye results with each image's serial result."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np

import deepface
from deepface.models.face_detection.opencv.OpenCv import OpenCvClient


cv2.setNumThreads(1)
source = cv2.imread(str(Path(deepface.__file__).parent.parent / "tests/unit/dataset/img1.jpg"))
assert source is not None
images = [
    cv2.resize(source, None, fx=0.25, fy=0.25),
    cv2.resize(source, None, fx=0.35, fy=0.35),
    np.zeros((128, 128, 3), np.uint8),
]
client = OpenCvClient()
face_crops = []
for image in images[:2]:
    area = client.detect_faces(image)[0]
    face_crops.append(image[area.y : area.y + area.h, area.x : area.x + area.w])
face_crops.append(images[2])
results = []
for operation, inputs in [("faces", images), ("eyes", face_crops)]:

    def run(index):
        try:
            if operation == "eyes":
                return client.find_eyes(inputs[index])
            return sorted(
                (int(f.x), int(f.y), int(f.w), int(f.h), f.left_eye, f.right_eye)
                for f in client.detect_faces(inputs[index])
            )
        except Exception as exc:
            return type(exc).__name__ + ": " + str(exc)

    expected = [run(i) for i in range(len(inputs))]
    assert expected[0] and expected[1]
    indices = [i % len(inputs) for i in range(90)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(run, indices))
    wrong = [(i, output) for i, output in zip(indices, outputs) if output != expected[i]]
    summary = {
        "operation": operation,
        "opencv": cv2.__version__,
        "calls": len(indices),
        "incorrect": len(wrong),
        "serial_results": expected,
        "examples": wrong[:3],
    }
    results.append(summary)
    print(json.dumps(summary), flush=True)
if "--baseline" in sys.argv:
    assert any(result["incorrect"] for result in results), "Native overlap did not reproduce"
else:
    assert all(result["incorrect"] == 0 for result in results)
