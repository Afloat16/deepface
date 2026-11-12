"""Bulk downloads prepare the same nested Buffalo_L cache that inference consumes."""

from pathlib import Path

import numpy as np
import pytest

from deepface.commons import backend_utils, weight_utils
from deepface.models.facial_recognition.onnx import Buffalo_L
from deepface.modules import modeling
from deepface.modules.exceptions import UnimplementedError


def only_buffalo(task, model_name):
    if task == "facial_recognition" and model_name == "Buffalo_L":
        return Buffalo_L.Buffalo_L
    raise UnimplementedError("Other downloads are outside this test")


def test_other_onnx_bulk_download_targets_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: "onnx")
    downloads = []
    monkeypatch.setattr(
        weight_utils,
        "download_weights_if_necessary",
        lambda **kwargs: downloads.append(kwargs),
    )

    weight_utils.download_all_models_in_one_shot()

    names = {item["file_name"] for item in downloads}
    assert {
        "facenet_weights.onnx",
        "facenet512_weights.onnx",
        "ghostfacenet_v1.onnx",
    } <= names
    archives = [item for item in downloads if item["compress_type"] == "bz2"]
    assert archives
    assert all(item["file_name"].endswith(".dat") for item in archives)


@pytest.mark.parametrize("backend", ["tensorflow", "pytorch", "onnx"])
def test_bulk_download_uses_buffalo_cache_path(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: backend)
    monkeypatch.setattr(modeling, "get_model_class", only_buffalo)
    monkeypatch.setattr(weight_utils, "ensure_source_is_reachable", lambda url: None)
    requests = []

    def download(url, output, quiet):
        requests.append(url)
        Path(output).write_bytes(b"cached model")
        return output

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    weight_utils.download_all_models_in_one_shot()

    expected = tmp_path / ".deepface" / "weights" / "buffalo_l" / "webface_r50.onnx"
    assert expected.read_bytes() == b"cached model"
    assert requests == [Buffalo_L.WEIGHTS_URL]

    weight_utils.download_all_models_in_one_shot()
    assert requests == [Buffalo_L.WEIGHTS_URL]


def test_prefetched_buffalo_loads_without_network(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    # A small graph verifies cache lookup and loading, not recognition accuracy.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ReduceMean", ["input"], ["output"], axes=[2, 3], keepdims=0
            )
        ],
        "cached-buffalo-fixture",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 3, 112, 112]
            )
        ],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = 7
    contents = model.SerializeToString()

    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    monkeypatch.setattr(backend_utils, "get_backend_engine", lambda: "onnx")
    monkeypatch.setattr(modeling, "get_model_class", only_buffalo)
    monkeypatch.setattr(weight_utils, "ensure_source_is_reachable", lambda url: None)

    def download(url, output, quiet):
        Path(output).write_bytes(contents)
        return output

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    weight_utils.download_all_models_in_one_shot()

    def offline(*args, **kwargs):
        raise AssertionError("The prefetched model must not require a network request")

    monkeypatch.setattr(weight_utils, "ensure_source_is_reachable", offline)
    monkeypatch.setattr(weight_utils.gdown, "download", offline)
    session = Buffalo_L.load_model()
    output = session.run(None, {"input": np.ones((1, 3, 112, 112), dtype=np.float32)})[
        0
    ]
    np.testing.assert_array_equal(output, np.ones((1, 3), dtype=np.float32))
