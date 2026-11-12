"""SFace batches preserve single-face output while bounding inference batches."""

from types import SimpleNamespace

import numpy as np
import pytest

from deepface.commons import weight_utils
from deepface.models.facial_recognition.onnx import SFace

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def client(tmp_path, monkeypatch):
    # A small graph with SFace's fixed exported batch size and embedding dimensions.
    weights = np.random.default_rng(0).normal(size=(3, 128)).astype(np.float32)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            onnx.helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            onnx.helper.make_node("MatMul", ["flat", "weights"], ["embedding"]),
        ],
        "sface-batch-fixture",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 3, 112, 112]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "embedding", onnx.TensorProto.FLOAT, [1, 128]
            )
        ],
        [onnx.numpy_helper.from_array(weights, "weights")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = 7
    path = tmp_path / "sface.onnx"
    path.write_bytes(model.SerializeToString())
    monkeypatch.setattr(
        weight_utils, "download_weights_if_necessary", lambda **kwargs: str(path)
    )

    client = SFace.SFaceClient()
    session = client.model
    calls = []

    def run(outputs, inputs):
        calls.append(inputs["input"].shape[0])
        return session.run(outputs, inputs)

    client.model = SimpleNamespace(run=run, get_inputs=session.get_inputs)
    return client, calls


@pytest.mark.parametrize("count", [1, 2, 8, 16, 17, 33])
def test_batched_output_matches_individual_faces(client, count):
    model, calls = client
    images = np.random.default_rng(1).integers(
        0, 256, (count, 112, 112, 3), dtype=np.uint8
    )
    # Exercise normalized, non-contiguous BGR input as returned by the preprocessing path.
    images = (images.astype(np.float32) / 255)[:, :, ::-1]
    expected = np.concatenate([model.predict(image) for image in images])
    calls.clear()

    actual = model.predict(images)

    assert actual.shape == (count, 128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
    assert sum(calls) == count
    assert max(calls) <= 16
    if count >= 16:
        assert len(calls) < count
    else:
        assert calls == [1] * count


def test_forward_keeps_single_and_batch_return_shapes(client):
    model, _ = client
    image = np.full((112, 112, 3), 0.5, dtype=np.float32)
    single = model.forward(image)
    batch = model.forward(np.stack([image, image]))
    assert len(single) == 128
    assert len(batch) == 2
    np.testing.assert_allclose(batch, [single, single])


@pytest.mark.parametrize("shape", [(0, 112, 112, 3), (112, 112), (1, 1, 112, 112, 3)])
def test_invalid_batch_shapes_still_raise(client, shape):
    model, calls = client
    with pytest.raises(ValueError, match="Input image must"):
        model.predict(np.zeros(shape, dtype=np.float32))
    assert not calls
