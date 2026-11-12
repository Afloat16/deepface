"""Repeated registration preserves existing identities in an actual local Qdrant store."""

import numpy as np
import pytest

from deepface.modules.database.qdrant import QdrantClient


@pytest.fixture
def gallery(tmp_path):
    sdk = pytest.importorskip("qdrant_client")
    connection = sdk.QdrantClient(path=str(tmp_path / "gallery"))
    for normalized in (False, True):
        connection.create_collection(
            collection_name=f"embeddings_facenet_skip_aligned_{'norm' if normalized else 'raw'}",
            vectors_config=sdk.models.VectorParams(
                size=3,
                distance=(
                    sdk.models.Distance.COSINE
                    if normalized
                    else sdk.models.Distance.EUCLID
                ),
            ),
        )
    client = QdrantClient(connection=connection)
    try:
        yield client
    finally:
        client.close()


def record(value, name, normalized=False):
    return {
        "img_name": name,
        "face": np.full((2, 2, 3), value, dtype=np.float32),
        "embedding": [float(value), 0.25, 0.5],
        "model_name": "Facenet",
        "detector_backend": "skip",
        "aligned": True,
        "l2_normalized": normalized,
    }


def entries(client, normalized=False):
    return client.fetch_all_embeddings(
        model_name="Facenet",
        detector_backend="skip",
        aligned=True,
        l2_normalized=normalized,
        batch_size=1,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_repeated_registration_does_not_replace_identity(gallery, normalized):
    assert gallery.insert_embeddings([record(1, "original.jpg", normalized)]) == 1
    original = entries(gallery, normalized)

    inserted = gallery.insert_embeddings([record(1, "renamed.jpg", normalized)])
    assert entries(gallery, normalized) == original
    assert inserted == 0


@pytest.mark.parametrize("batch_size", [1, 2, 3, 100])
def test_input_duplicates_preserve_first_identity_across_batches(gallery, batch_size):
    records = [
        record(1, "first.jpg"),
        record(2, "second.jpg"),
        record(1, "duplicate.jpg"),
        record(3, "third.jpg"),
        record(2, "another-duplicate.jpg"),
    ]
    assert gallery.insert_embeddings(records, batch_size=batch_size) == 3
    assert {e["img_name"] for e in entries(gallery)} == {
        "first.jpg",
        "second.jpg",
        "third.jpg",
    }


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_mixed_existing_and_new_records_count_only_new_points(gallery, batch_size):
    assert gallery.insert_embeddings([record(1, "existing.jpg")]) == 1
    existing = entries(gallery)[0]

    assert (
        gallery.insert_embeddings(
            [
                record(1, "overwritten.jpg"),
                record(2, "new.jpg"),
                record(2, "repeat.jpg"),
            ],
            batch_size=batch_size,
        )
        == 1
    )
    current = entries(gallery)
    assert existing in current
    assert {e["img_name"] for e in current} == {"existing.jpg", "new.jpg"}


def test_duplicates_are_scoped_to_the_collection(gallery):
    assert gallery.insert_embeddings([record(1, "raw.jpg")]) == 1
    assert gallery.insert_embeddings([record(1, "normalized.jpg", True)]) == 1
    assert len(entries(gallery)) == len(entries(gallery, True)) == 1


def test_different_faces_with_the_same_embedding_keep_distinct_ids(gallery):
    first, second = record(1, "first.jpg"), record(2, "second.jpg")
    second["embedding"] = first["embedding"]
    assert gallery.insert_embeddings([first, second]) == 2
    assert len({e["id"] for e in entries(gallery)}) == 2


def test_retry_remains_idempotent_after_reopening(tmp_path):
    sdk = pytest.importorskip("qdrant_client")
    path = str(tmp_path / "persistent-gallery")
    connection = sdk.QdrantClient(path=path)
    connection.create_collection(
        collection_name="embeddings_facenet_skip_aligned_raw",
        vectors_config=sdk.models.VectorParams(
            size=3, distance=sdk.models.Distance.EUCLID
        ),
    )
    client = QdrantClient(connection=connection)
    try:
        assert client.insert_embeddings([record(1, "original.jpg")]) == 1
        original = entries(client)
    finally:
        client.close()

    reopened = QdrantClient(connection=sdk.QdrantClient(path=path))
    try:
        assert reopened.insert_embeddings([record(1, "retry.jpg")]) == 0
        assert entries(reopened) == original
    finally:
        reopened.close()
