"""Operation-local Pinecone indexes are released on every exit path."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from deepface.modules.database.pinecone import PineconeClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


def build_client():
    client = PineconeClient.__new__(PineconeClient)
    client.client = Mock()
    client.client.has_index.return_value = True
    index = client.client.Index.return_value
    index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(id="person", score=0.0, metadata={"img_name": "person.png"})]
    )
    index.fetch.return_value = {
        "vectors": {"person": {"values": [1.0, 0.0], "metadata": {"img_name": "person.png"}}}
    }
    index.list.return_value = [["person"]]
    return client, index


def invoke(client, operation):
    if operation == "insert":
        return client.insert_embeddings(
            [dict(CRITERIA, face=np.zeros((2, 2, 3)), embedding=[1.0, 0.0], img_name="person.png")]
        )
    if operation == "search":
        return client.search_by_vector([1.0, 0.0], **CRITERIA)
    if operation == "fetch":
        return client.fetch_embedding("person", **CRITERIA)
    return client.fetch_all_embeddings(**CRITERIA)


@pytest.mark.parametrize("operation", ["insert", "search", "fetch", "list"])
def test_success_closes_only_the_operation_index(operation):
    client, index = build_client()

    result = invoke(client, operation)

    index.close.assert_called_once_with()
    client.client.close.assert_not_called()
    if operation == "insert":
        assert result == 1
        assert index.upsert.call_args.kwargs["vectors"][0]["values"] == [1.0, 0.0]
    elif operation == "fetch":
        assert result["id"] == "person"
        assert result["embedding"] == [1.0, 0.0]
    else:
        assert result[0]["id"] == "person"
        assert result[0]["img_name"] == "person.png"


@pytest.mark.parametrize(
    "operation,method",
    [
        ("insert", "upsert"),
        ("search", "query"),
        ("fetch", "fetch"),
        ("list", "list"),
        ("list", "fetch"),
    ],
)
def test_failure_closes_the_index_and_preserves_the_exception(operation, method):
    client, index = build_client()
    error = RuntimeError("request failed")
    getattr(index, method).side_effect = error

    with pytest.raises(RuntimeError) as caught:
        invoke(client, operation)

    assert caught.value is error
    assert caught.value.__traceback__ is not None
    index.close.assert_called_once_with()
    client.client.close.assert_not_called()


@pytest.mark.parametrize("operation", ["search", "fetch", "list"])
def test_empty_results_still_close_the_index(operation):
    client, index = build_client()
    index.query.return_value = SimpleNamespace(matches=[])
    index.fetch.return_value = {"vectors": {}}
    index.list.return_value = []

    assert invoke(client, operation) == (None if operation == "fetch" else [])

    index.close.assert_called_once_with()
    client.client.close.assert_not_called()


def test_missing_index_does_not_create_an_operation_client():
    client, _ = build_client()
    client.client.has_index.return_value = False

    assert client.fetch_embedding("person", **CRITERIA) is None

    client.client.Index.assert_not_called()
    client.client.close.assert_not_called()


def test_adapter_close_releases_the_control_plane_client():
    client, _ = build_client()

    client.close()

    client.client.close.assert_called_once_with()
    client.client.Index.assert_not_called()


def test_adapter_close_accepts_legacy_clients_without_close():
    client = PineconeClient.__new__(PineconeClient)
    client.client = Mock(spec=[])

    assert client.close() is None
