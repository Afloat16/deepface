"""Real OpenCV/SFace coverage for cached gallery entries with no detected face."""

from pathlib import Path
import pickle
import shutil

import cv2
import numpy as np
import pytest

from deepface.modules import recognition


@pytest.mark.parametrize("missing", ["first", "last", "all"])
@pytest.mark.parametrize("similarity_search", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_sface_find_preserves_faceless_gallery(
    tmp_path, missing, similarity_search, batched
):
    source = Path(__file__).resolve().parents[1] / "unit" / "dataset" / "img1.jpg"
    faceless = str(tmp_path / "no-face.png")
    assert cv2.imwrite(faceless, np.zeros((224, 224, 3), dtype=np.uint8))
    valid = str(tmp_path / "face.jpg")
    if missing != "all":
        shutil.copyfile(source, valid)
    options = dict(
        img_path=str(source),
        db_path=str(tmp_path),
        model_name="SFace",
        detector_backend="opencv",
        distance_metric="euclidean",
        silent=True,
    )

    # Build real embeddings and the real no-face cache row before controlling row order.
    recognition.find(**options, similarity_search=True)
    cache = next(tmp_path.glob("ds_model_*.pkl"))
    gallery = pickle.loads(cache.read_bytes())
    assert len(gallery) == (1 if missing == "all" else 2)
    assert (
        next(row for row in gallery if row["identity"] == faceless)["embedding"] is None
    )
    gallery.sort(
        key=lambda row: row["embedding"] is not None, reverse=missing == "last"
    )
    cache.write_bytes(pickle.dumps(gallery))

    result = recognition.find(
        **options,
        batched=batched,
        similarity_search=similarity_search,
        refresh_database=False,
    )[0]
    rows = result if batched else result.to_dict("records")
    expected = [] if missing == "all" else [valid]
    if similarity_search:
        expected.append(faceless)
    assert [row["identity"] for row in rows] == expected
    for row in rows:
        if row["identity"] == faceless:
            assert row["distance"] == float("inf")
            assert row["target_w"] == row["target_h"] == 0
        else:
            assert row["distance"] == pytest.approx(0, abs=1e-6)
            assert row["target_w"] > 0 and row["target_h"] > 0
    assert pickle.loads(cache.read_bytes()) == gallery
