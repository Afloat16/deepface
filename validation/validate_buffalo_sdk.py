"""Compare the Buffalo adapter with the installed InsightFace preprocessing path."""
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from insightface.model_zoo.arcface_onnx import ArcFaceONNX

from deepface.models.facial_recognition.onnx.Buffalo_L import Buffalo_L
from deepface.modules import preprocessing


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "channel-means.onnx"
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["output"], axes=[2, 3], keepdims=0)],
        "channel-means",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 112, 112])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sdk = ArcFaceONNX(model_file=str(path), session=session)
    client = Buffalo_L.__new__(Buffalo_L)
    client.model = sdk

    raw = np.empty((112, 112, 3), dtype=np.uint8)
    raw[:] = [32, 96, 224]
    second = np.empty_like(raw)
    second[:] = [200, 80, 20]
    for faces in ([raw], [raw, second]):
        inputs = np.concatenate([preprocessing.resize_image(face, (112, 112)) for face in faces])
        expected = np.concatenate([sdk.get_feat(face) for face in faces])
        result = np.asarray(client.forward(inputs))
        result = result.reshape(len(faces), -1)
        print(json.dumps({"faces": len(faces), "expected": expected.tolist(), "actual": result.tolist()}))
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)
    print("InsightFace 0.7.3 native ONNX Runtime single and batch preprocessing parity passed")
