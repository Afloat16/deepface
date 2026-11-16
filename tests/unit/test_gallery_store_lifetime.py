"""Gallery stores are closed even when an exception keeps their frames alive."""

import ftplib
from unittest.mock import Mock, patch

import numpy as np
import pytest

from deepface.modules import filestore, modeling, recognition
from deepface.modules.exceptions import EmptyDatasource, ImgNotFound, PathNotFound

CACHE_NAME = "ds_model_sface_detector_skip_aligned_normalization_base_expand_0.npz"
QUERY = np.full((8, 8, 3), 128, dtype=np.uint8)


def cache_bytes(identity="person.png"):
    return recognition.__serialize_npz(
        [
            {
                "identity": identity,
                "hash": "cached",
                "embedding": [1.0, 0.0],
                "target_x": 0,
                "target_y": 0,
                "target_w": 8,
                "target_h": 8,
            }
        ]
    )


def find(gallery="ftp://gallery.example/", **kwargs):
    options = dict(
        img_path=QUERY,
        db_path=gallery,
        model_name="SFace",
        detector_backend="skip",
        enforce_detection=False,
        align=True,
        silent=True,
        return_type="dict",
    )
    options.update(kwargs)
    return recognition.find(**options)


@pytest.fixture
def store():
    result = Mock()
    result.join.return_value = "ftp://gallery.example/" + CACHE_NAME
    result.exists.return_value = True
    result.read_bytes.return_value = cache_bytes()
    result.list_images.return_value = []
    return result


@pytest.fixture
def inference():
    model = Mock(input_shape=(8, 8), output_shape=2)
    model.forward.return_value = [1.0, 0.0]
    with patch.object(modeling, "build_model", return_value=model):
        yield model


@pytest.mark.parametrize(
    "failure,error",
    [("query", ImgNotFound), ("empty_gallery", EmptyDatasource), ("cache", ValueError)],
)
def test_find_closes_store_while_failure_traceback_is_retained(store, failure, error):
    options = {}
    if failure == "query":
        image_result = (None, "missing.png")
    else:
        image_result = (QUERY, None)
    if failure == "cache":
        store.read_bytes.return_value = b"invalid NPZ archive"
        options["refresh_database"] = False

    with patch.object(recognition, "build_file_store", return_value=store), patch.object(
        recognition.image_utils, "load_image", return_value=image_result
    ):
        with pytest.raises(error) as caught:
            find(**options)

    assert caught.value.__traceback__ is not None
    store.close.assert_called_once_with()


def test_failed_ftp_validation_closes_the_opened_connection():
    connection = Mock(spec=ftplib.FTP)
    connection.cwd.side_effect = ftplib.error_perm("550 missing directory")

    with patch.object(filestore.ftplib, "FTP", return_value=connection):
        with pytest.raises(PathNotFound) as caught:
            filestore.build_file_store("ftp://gallery.example/missing")

    assert caught.value.__traceback__ is not None
    connection.connect.assert_called_once()
    connection.close.assert_called_once_with()


def test_failed_s3_validation_closes_the_client():
    store = filestore.S3FileStore.__new__(filestore.S3FileStore)
    store.bucket = "gallery"
    store.client = Mock()
    store.client.head_bucket.side_effect = RuntimeError("access denied")

    with patch.object(filestore, "S3FileStore", return_value=store):
        with pytest.raises(PathNotFound) as caught:
            filestore.build_file_store("s3://gallery")

    assert caught.value.__traceback__ is not None
    store.client.close.assert_called_once_with()


@pytest.mark.parametrize("return_type", ["dict", "pandas"])
def test_success_closes_store_before_query_inference(store, inference, return_type):
    def represent(_images):
        store.close.assert_called_once_with()
        return [1.0, 0.0]

    inference.forward.side_effect = represent
    with patch.object(recognition, "build_file_store", return_value=store):
        results = find(refresh_database=False, return_type=return_type)

    rows = results[0] if return_type == "dict" else results[0].to_dict("records")
    assert rows[0]["identity"] == "person.png"
    assert rows[0]["distance"] == 0
    store.close.assert_called_once_with()


@pytest.mark.parametrize("return_type", ["dict", "pandas"])
def test_local_gallery_preserves_results(tmp_path, inference, return_type):
    (tmp_path / CACHE_NAME).write_bytes(cache_bytes("local-person.png"))

    results = find(str(tmp_path), refresh_database=False, return_type=return_type)

    rows = results[0] if return_type == "dict" else results[0].to_dict("records")
    assert rows[0]["identity"] == "local-person.png"
    assert rows[0]["distance"] == 0


def test_ftp_close_releases_and_clears_the_cached_connection():
    store = filestore.FtpFileStore("ftp://gallery.example/")
    connection = Mock(spec=ftplib.FTP)
    store.ftp = connection

    store.close()
    store.close()

    connection.close.assert_called_once_with()
    assert store.ftp is None


def test_ftp_close_before_connecting_is_safe():
    store = filestore.FtpFileStore("ftp://gallery.example/")

    assert store.close() is None


def test_s3_close_delegates_to_the_client():
    store = filestore.S3FileStore.__new__(filestore.S3FileStore)
    store.client = Mock()

    store.close()

    store.client.close.assert_called_once_with()
