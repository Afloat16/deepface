"""Faceless gallery entries retain infinite distance without breaking recognition."""

from copy import deepcopy
import pickle

import cv2
import numpy as np
import pytest

from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import detection, modeling, recognition, verification
from deepface.modules.exceptions import SpoofDetected


class ColorModel(FacialRecognition):
    input_shape = (8, 8)
    output_shape = 3

    def forward(self, img):
        embeddings = img.mean(axis=(1, 2)).tolist()
        return embeddings[0] if len(embeddings) == 1 else embeddings


@pytest.mark.parametrize("missing", ["first", "last", "all", "none"])
@pytest.mark.parametrize("similarity_search", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "distance_metric", ["euclidean", "euclidean_l2", "cosine", "angular"]
)
def test_find_preserves_missing_embeddings(
    tmp_path, monkeypatch, missing, similarity_search, batched, distance_metric
):
    image = np.full((8, 8, 3), [51, 102, 153], dtype=np.uint8)
    gallery_path = str(tmp_path / "gallery.png")
    assert cv2.imwrite(gallery_path, image)
    valid = {
        "identity": gallery_path,
        "hash": "cached",
        "embedding": [0.2, 0.4, 0.6],
        "target_x": 0,
        "target_y": 0,
        "target_w": 8,
        "target_h": 8,
    }
    invalid = dict(
        valid,
        identity=str(tmp_path / "no-face.png"),
        hash="faceless",
        embedding=None,
        target_x=17,
        target_w=0,
    )
    galleries = {
        "first": [invalid, valid],
        "last": [valid, invalid],
        "all": [invalid],
        "none": [valid],
    }
    cache = (
        tmp_path
        / "ds_model_facenet512_detector_skip_unaligned_normalization_base_expand_0.pkl"
    )
    cache.write_bytes(pickle.dumps(galleries[missing]))
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())
    source_region = detection.extract_faces(
        image, detector_backend="skip", align=False, color_face="bgr"
    )[0]["facial_area"]

    results = recognition.find(
        image,
        str(tmp_path),
        model_name="Facenet512",
        detector_backend="skip",
        align=False,
        distance_metric=distance_metric,
        threshold=0.01,
        refresh_database=False,
        batched=batched,
        similarity_search=similarity_search,
        silent=True,
    )

    assert len(results) == 1
    rows = results[0] if batched else results[0].to_dict("records")
    expected = [] if missing == "all" else [gallery_path]
    if similarity_search and missing != "none":
        expected.append(invalid["identity"])
    assert [item["identity"] for item in rows] == expected
    for item in rows:
        assert "embedding" not in item
        if item["identity"] == gallery_path:
            assert item["distance"] == pytest.approx(0, abs=2e-5)
            assert item["hash"] == "cached"
            assert item["target_x"] == 0
        else:
            assert item["distance"] == float("inf")
            assert item["hash"] == "faceless"
            assert item["target_x"] == 17
            assert item["target_w"] == 0
        for dimension in ["x", "y", "w", "h"]:
            assert item[f"source_{dimension}"] == source_region[dimension]
    assert pickle.loads(cache.read_bytes()) == galleries[missing]


@pytest.mark.parametrize("gallery", [[], [{"embedding": None}]])
@pytest.mark.parametrize("similarity_search", [False, True])
def test_missing_gallery_preserves_query_antispoofing(gallery, similarity_search):
    source = {"face": np.zeros((8, 8, 3)), "facial_area": {}, "is_real": False}
    with pytest.raises(SpoofDetected):
        recognition.find_batched(
            gallery,
            [source],
            anti_spoofing=True,
            threshold=0.5,
            similarity_search=similarity_search,
        )


@pytest.mark.parametrize("similarity_search", [False, True])
@pytest.mark.parametrize("k", [None, 0, 1, 2, 3, 4, 5])
def test_missing_gallery_keeps_query_order_and_top_k_metadata(
    monkeypatch, similarity_search, k
):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())
    gallery = [
        {"identity": "no-face", "embedding": None, "target_x": 0, "note": "missing"},
        {"identity": "second", "embedding": [0.8, 0.6, 0.4], "target_x": 20},
        {"identity": "also-no-face", "embedding": None, "target_x": 30},
        {"identity": "first", "embedding": [0.2, 0.4, 0.6], "target_x": 10},
    ]
    original = deepcopy(gallery)
    sources = [
        {
            "face": np.full((8, 8, 3), color, dtype=np.uint8),
            "facial_area": {"x": x, "y": 0, "w": 8, "h": 8},
        }
        for x, color in [(1, [51, 102, 153]), (2, [204, 153, 102])]
    ]
    result = recognition.find_batched(
        gallery,
        sources,
        model_name="Facenet512",
        distance_metric="euclidean",
        threshold=0.01,
        similarity_search=similarity_search,
        k=k,
    )
    assert len(result) == 2
    for x, (rows, closest) in enumerate(zip(result, ["first", "second"]), start=1):
        valid = [closest]
        if similarity_search:
            valid.append("second" if closest == "first" else "first")
        expected_size = len(valid) + (2 if similarity_search else 0)
        if k is not None:
            expected_size = min(expected_size, k)
        assert len(rows) == expected_size
        assert [row["identity"] for row in rows[:2]] == valid[:expected_size]
        for row in rows:
            source = next(
                item for item in gallery if item["identity"] == row["identity"]
            )
            assert row["target_x"] == source["target_x"]
            assert row["note"] == source.get("note")
            assert row["source_x"] == x
            assert "embedding" not in row
            if source["embedding"] is None:
                assert row["distance"] == float("inf")
            else:
                assert np.isfinite(row["distance"])
        if k is None and similarity_search:
            assert {row["identity"] for row in rows[2:]} == {"no-face", "also-no-face"}
    assert gallery == original


@pytest.mark.parametrize("missing", ["mixed", "all", "empty"])
@pytest.mark.parametrize("similarity_search", [False, True])
def test_custom_metric_receives_only_valid_embeddings(
    monkeypatch, missing, similarity_search
):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())
    gallery = [{"identity": "no-face", "embedding": None}]
    if missing == "mixed":
        gallery += [
            {"identity": "valid", "embedding": [0.2, 0.4, 0.6]},
            {"identity": "missing-key"},
        ]
    elif missing == "empty":
        gallery = []
    sources = [
        {
            "face": np.full((8, 8, 3), color, dtype=np.uint8),
            "facial_area": {"x": x, "y": 0, "w": 8, "h": 8},
        }
        for x, color in [(1, [51, 102, 153]), (2, [204, 153, 102])]
    ]
    calls = []

    def metric(gallery_embeddings, query_embeddings):
        calls.append((gallery_embeddings.copy(), query_embeddings.copy()))
        return np.linalg.norm(
            query_embeddings[:, None] - gallery_embeddings[None, :], axis=2
        )

    result = recognition.find_batched(
        gallery,
        sources,
        model_name="Facenet512",
        distance_metric=metric,
        threshold=0.01,
        similarity_search=similarity_search,
    )
    assert len(result) == 2
    assert len(calls) == (1 if missing == "mixed" else 0)
    if calls:
        assert calls[0][0].shape == (1, 3)
        assert calls[0][1].shape == (2, 3)
        np.testing.assert_allclose(calls[0][0], [[0.2, 0.4, 0.6]])
    for i, rows in enumerate(result):
        if similarity_search:
            assert len(rows) == len(gallery)
            if missing == "mixed":
                assert rows[0]["identity"] == "valid"
                assert np.isfinite(rows[0]["distance"])
                assert {row["identity"] for row in rows[1:]} == {
                    "no-face",
                    "missing-key",
                }
                assert all(row["distance"] == float("inf") for row in rows[1:])
            else:
                assert all(row["distance"] == float("inf") for row in rows)
        else:
            assert [row["identity"] for row in rows] == (
                ["valid"] if missing == "mixed" and i == 0 else []
            )


@pytest.mark.parametrize("gallery", [[], [{"embedding": None}], [{"embedding": [1.0]}]])
def test_empty_sources_return_no_results(gallery):
    def metric(*args):
        pytest.fail("Distance calculation must not run without queries")

    assert recognition.find_batched(gallery, [], distance_metric=metric) == []


@pytest.mark.parametrize("missing", ["all", "none", "mixed", "empty"])
@pytest.mark.parametrize("threshold", [None, 0.0])
@pytest.mark.parametrize("similarity_search", [False, True])
def test_direct_batched_find_default_and_zero_threshold(
    monkeypatch, missing, threshold, similarity_search
):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: ColorModel())
    valid = {"identity": "valid", "embedding": [0.2, 0.4, 0.6]}
    invalid = {"identity": "no-face", "embedding": None}
    gallery = {
        "all": [invalid],
        "none": [valid],
        "mixed": [invalid, valid],
        "empty": [],
    }[missing]
    source = {
        "face": np.full((8, 8, 3), [51, 102, 153], dtype=np.uint8),
        "facial_area": {"x": 0, "y": 0, "w": 8, "h": 8},
    }
    result = recognition.find_batched(
        gallery,
        [source],
        model_name="Facenet512",
        distance_metric="euclidean",
        threshold=threshold,
        similarity_search=similarity_search,
    )
    expected = ["valid"] if missing in ["none", "mixed"] else []
    if similarity_search and missing in ["all", "mixed"]:
        expected.append("no-face")
    assert len(result) == 1
    assert [row["identity"] for row in result[0]] == expected
    expected_threshold = (
        float("inf")
        if similarity_search
        else (
            verification.find_threshold("Facenet512", "euclidean")
            if threshold is None
            else threshold
        )
    )
    assert all(row["threshold"] == expected_threshold for row in result[0])
