"""Database identities survive Struct's double-only number representation."""

import numpy as np
import pytest

grpc = pytest.importorskip("grpc")
json_format = pytest.importorskip("google.protobuf.json_format")
struct_pb2 = pytest.importorskip("google.protobuf.struct_pb2")

try:
    from deepface.api.proto import deepface_pb2, deepface_pb2_grpc
    from deepface.api.src.grpc_server import create_server, to_struct
except ImportError:
    pytest.skip(
        "gRPC stubs not generated, run `make grpc` first", allow_module_level=True
    )

from deepface.modules import datastore
from deepface.modules.database.sqlite import SqliteClient


@pytest.mark.parametrize(
    "identity",
    [
        2**53,
        2**53 + 1,
        -(2**53 + 1),
        2**63 - 1,
        np.int64(2**63 - 1),
        np.uint64(2**64 - 1),
    ],
)
def test_large_integer_identity_round_trips_exactly(identity):
    response = to_struct({"results": [[{"id": identity}]]})
    restored = struct_pb2.Struct.FromString(response.SerializeToString())
    result = json_format.MessageToDict(restored)
    assert result["results"][0][0]["id"] == str(identity)


def test_numpy_integer_arrays_do_not_lose_bits():
    identities = np.array([2**53 + 1, 2**63 - 1], dtype=np.int64)
    result = json_format.MessageToDict(to_struct({"ids": identities}))
    assert result["ids"] == [str(value) for value in identities]


def test_safe_numbers_and_other_json_types_are_preserved():
    original = {
        "small": 42,
        "min": -(2**53 - 1),
        "max": 2**53 - 1,
        "distance": 0.125,
        "large_float": 1e20,
        "verified": True,
        "missing": None,
        "label": "person",
        "embedding": np.array([0.25, -0.5], dtype=np.float32),
    }
    result = json_format.MessageToDict(to_struct(original))
    assert result == {**original, "embedding": [0.25, -0.5]}
    assert isinstance(result["small"], float)
    assert isinstance(result["large_float"], float)
    assert result["verified"] is True


def test_sqlite_search_ids_survive_grpc_transport(tmp_path, monkeypatch):
    path = str(tmp_path / "gallery.sqlite")
    client = SqliteClient(connection_details=path)
    identities = [2**53 + 1, 2**63 - 1]
    try:
        client.insert_embeddings(
            [
                {
                    "img_name": f"person-{index}.jpg",
                    "face": np.full((2, 2, 3), index, dtype=np.float32),
                    "embedding": embedding,
                    "model_name": "Facenet",
                    "detector_backend": "skip",
                    "aligned": True,
                    "l2_normalized": False,
                }
                for index, embedding in enumerate(([1.0, 0.0], [0.0, 1.0]))
            ]
        )
        for original_id, identity in enumerate(identities, start=1):
            client.conn.execute(
                "UPDATE embeddings SET id = ? WHERE id = ?", (identity, original_id)
            )
        client.conn.commit()
    finally:
        client.close()

    # Inference is isolated; the database search, protobuf and gRPC transport are real.
    monkeypatch.setattr(
        datastore, "represent", lambda **kwargs: [{"embedding": [1.0, 0.0]}]
    )
    monkeypatch.setenv("DEEPFACE_DATABASE_TYPE", "sqlite")
    monkeypatch.setenv("DEEPFACE_CONNECTION_DETAILS", path)
    monkeypatch.delenv("DEEPFACE_FACE_RECOGNITION_MODELS", raising=False)
    monkeypatch.delenv("DEEPFACE_FACE_DETECTION_MODELS", raising=False)
    monkeypatch.delenv("DEEPFACE_AUTH_TOKEN", raising=False)
    server, port = create_server(port=0)
    server.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = deepface_pb2_grpc.DeepFaceServiceStub(channel)
            response = stub.Search(
                deepface_pb2.SearchRequest(
                    img=deepface_pb2.Image(path="inference-is-isolated"),
                    model_name="Facenet",
                    detector_backend="skip",
                    distance_metric="euclidean",
                    similarity_search=True,
                ),
                timeout=10,
            )
        results = json_format.MessageToDict(response.result)["results"][0]
        assert [item["id"] for item in results] == [
            str(identity) for identity in identities
        ]
        assert [item["img_name"] for item in results] == [
            "person-0.jpg",
            "person-1.jpg",
        ]
    finally:
        server.stop(None).wait(timeout=10)
