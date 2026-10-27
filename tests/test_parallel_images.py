import base64
import copy
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError


def encoded(seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (128, 160, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("right", [None, 2])
def test_parallel_decode_and_resize_preserve_pixels(right):
    obs = {
        "images": {
            "head_camera": encoded(0),
            "left_wrist_camera": encoded(1),
            "right_wrist_camera": encoded(right) if right is not None else None,
        },
        "prompt": "test",
        "state": [0.0] * 10,
    }
    serial = ObsPreprocessor(
        multiview=True, camera_layout=["a", "b", None if right is None else "c"], img_height=385, img_width=321
    )
    parallel = ObsPreprocessor(
        multiview=True, camera_layout=serial.camera_layout, img_height=385, img_width=321, decode_workers=2
    )
    try:
        for _ in range(3):
            expected = serial.preprocess(copy.deepcopy(obs))
            actual = parallel.preprocess(copy.deepcopy(obs))
            assert np.array_equal(np.asarray(expected["image"]), np.asarray(actual["image"]))
            assert actual["prompt"] == expected["prompt"]
            np.testing.assert_array_equal(actual["state"], expected["state"])
    finally:
        parallel.shutdown()
    assert parallel._decode_pool is None


def test_decode_errors_preserve_camera_priority_and_pool_recovers():
    parallel = ObsPreprocessor(
        multiview=True, camera_layout=["a", "b", "c"], img_height=384, img_width=320, decode_workers=2
    )
    try:
        with pytest.raises(ObsValidationError, match="head_camera"):
            parallel.preprocess({"images": {"head_camera": "bad", "left_wrist_camera": encoded(1)}})
        with pytest.raises(ObsValidationError, match="left_wrist_camera"):
            parallel.preprocess(
                {"images": {"head_camera": encoded(0), "left_wrist_camera": 7, "right_wrist_camera": "bad"}}
            )
        value = parallel.preprocess({"images": {"head_camera": encoded(0), "left_wrist_camera": encoded(1)}})
        assert value["image"].size == (320, 384)
    finally:
        parallel.shutdown()


def test_parallel_composition_preserves_missing_mask_and_shared_pil():
    image = Image.new("RGB", (32, 32), (45, 67, 123))
    frames = {"a": image, "b": image}
    args = dict(camera_layout=["a", "b", "c"], out_h=65, out_w=79, return_missing_mask=True)
    expected = assemble_multiview_layout(frames, **args)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = assemble_multiview_layout(frames, executor=executor, **args)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
