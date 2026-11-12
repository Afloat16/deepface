"""Real CPU autocast inference must cross the NumPy boundary successfully."""

from contextlib import nullcontext

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepface.models.demography.pytorch.Emotion import EmotionClient, EmotionModel
from deepface.models.facial_recognition.pytorch.DeepID import DeepIdClient, DeepIdNet
from deepface.models.spoofing.pytorch.FasNet import Fasnet
from deepface.models.spoofing.pytorch.FasNetBackbone import MiniFASNetV1SE, MiniFASNetV2


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def autocast(enabled):
    return torch.autocast("cpu", dtype=torch.bfloat16) if enabled else nullcontext()


@pytest.fixture
def recognition_client():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        client = DeepIdClient.__new__(DeepIdClient)
        client.device = torch.device("cpu")
        client.model = DeepIdNet().eval()
    return client


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_recognition_predict_and_forward(
    recognition_client, batch_size, mixed_precision
):
    images = np.random.default_rng(42).random((batch_size, 55, 47, 3), dtype=np.float32)
    with autocast(mixed_precision):
        with torch.no_grad():
            expected = recognition_client.model(recognition_client.to_tensor(images))
        assert expected.dtype == (torch.bfloat16 if mixed_precision else torch.float32)
        expected = expected.float().numpy()

        actual = recognition_client.predict(images)
        forwarded = recognition_client.forward(images)

    assert actual.dtype == np.float32
    assert actual.shape == (batch_size, 160)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        forwarded, expected[0] if batch_size == 1 else expected
    )


@pytest.fixture
def emotion_client():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        client = EmotionClient.__new__(EmotionClient)
        client.device = torch.device("cpu")
        client.model_name = "Emotion"
        client.model = EmotionModel().eval()
    return client


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_demography_single_and_batch_predictions(
    emotion_client, batch_size, mixed_precision
):
    images = np.random.default_rng(42).random(
        (batch_size, 224, 224, 3), dtype=np.float32
    )
    processed = np.stack([emotion_client._preprocess_image(image) for image in images])
    tensor = torch.from_numpy(processed[:, None])
    with autocast(mixed_precision):
        with torch.no_grad():
            expected = emotion_client.model(tensor)
        assert expected.dtype == (torch.bfloat16 if mixed_precision else torch.float32)
        expected = expected.float().numpy()
        actual = emotion_client.predict(images[0] if batch_size == 1 else images)

    assert actual.dtype == np.float32
    assert actual.shape == ((7,) if batch_size == 1 else (batch_size, 7))
    np.testing.assert_allclose(
        actual, expected[0] if batch_size == 1 else expected, rtol=1e-6
    )
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=3e-3)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_antispoofing_with_native_minifasnet_models(mixed_precision):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        client = Fasnet.__new__(Fasnet)
        client.device = torch.device("cpu")
        client.first_model = MiniFASNetV2(conv6_kernel=(5, 5)).eval()
        client.second_model = MiniFASNetV1SE(conv6_kernel=(5, 5)).eval()

    images = np.random.default_rng(42).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    # A full-frame ROI retains the complete image for both crop scales.
    tensor = torch.from_numpy(images.transpose(2, 0, 1)[None]).float()
    with autocast(mixed_precision):
        with torch.no_grad():
            first = torch.softmax(client.first_model(tensor), dim=1)
            second = torch.softmax(client.second_model(tensor), dim=1)
        assert first.dtype == (torch.bfloat16 if mixed_precision else torch.float32)
        expected = (
            first.float().numpy().astype(np.float64) + second.float().numpy()
        ) / 2
        actual = client.analyze(images, (0, 0, 80, 80))

    label = int(expected.argmax())
    assert actual[0] == (label == 1)
    assert actual[1] == expected[0, label]


@pytest.mark.parametrize(
    "output_dtype,expected_dtype",
    [
        (torch.float16, np.float16),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
        (torch.bfloat16, np.float32),
    ],
)
def test_numpy_compatible_output_dtypes_are_preserved(
    recognition_client, output_dtype, expected_dtype
):
    class OutputDtype(torch.nn.Module):
        def forward(self, tensor):
            return tensor.to(output_dtype)

    recognition_client.model = torch.nn.Sequential(
        recognition_client.model, OutputDtype()
    )
    image = np.random.default_rng(42).random((1, 55, 47, 3), dtype=np.float32)
    actual = recognition_client.predict(image)
    assert actual.dtype == expected_dtype
    with torch.no_grad():
        expected = (
            recognition_client.model(recognition_client.to_tensor(image))
            .double()
            .numpy()
        )
    np.testing.assert_array_equal(actual, expected)
