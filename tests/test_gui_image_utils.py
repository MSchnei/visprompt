import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage

from visprompt.gui.gui_image_utils import (
    get_segmentation_image_from_sam,
    get_segmentation_image_from_seggpt,
    numpy_array_to_qimage,
    qimage_to_numpy_array,
    transform_points,
)


@pytest.mark.parametrize(
    "points, scale_factor, x_offset, y_offset, expected",
    [
        # No change
        (
            [QPoint(0, 0), QPoint(1, 1), QPoint(2, 2)],
            1,
            0,
            0,
            [QPoint(0, 0), QPoint(1, 1), QPoint(2, 2)],
        ),
        # Scaling
        ([QPoint(10, 10), QPoint(20, 20)], 10, 0, 0, [QPoint(1, 1), QPoint(2, 2)]),
        # Translation
        ([QPoint(10, 10), QPoint(20, 20)], 1, 10, 10, [QPoint(0, 0), QPoint(10, 10)]),
        # Combined scaling and translation
        (
            [QPoint(110, 110), QPoint(220, 220)],
            10,
            100,
            100,
            [QPoint(1, 1), QPoint(12, 12)],
        ),
        # Negative scale
        (
            [QPoint(10, 10), QPoint(20, 20)],
            -1,
            0,
            0,
            [QPoint(-10, -10), QPoint(-20, -20)],
        ),
    ],
)
def test_transform_points_varied_cases(
    points, scale_factor, x_offset, y_offset, expected
):
    transformed = transform_points(points, scale_factor, x_offset, y_offset)
    assert all(
        t.x() == e.x() and t.y() == e.y() for t, e in zip(transformed, expected)
    ), "Points should be transformed as expected"


# Test for an empty list of points
def test_transform_points_empty_list():
    transformed = transform_points([], 5, 10, 10)
    assert transformed == [], "Transforming an empty list should return an empty list"


# Test for zero scale factor, expecting an error
def test_transform_points_zero_scale():
    points = [QPoint(10, 10), QPoint(20, 20)]
    with pytest.raises(ZeroDivisionError):
        transform_points(points, 0, 10, 10)


def create_test_qimage(width, height, format):
    image = QImage(width, height, format)
    # Fill the image with a solid color for simplicity
    color = QColor(128, 128, 128)  # Grey color
    image.fill(color)
    return image


@pytest.mark.parametrize(
    "image_format, expected_dtype, expected_channels",
    [
        (QImage.Format_RGB32, np.uint8, 3),  # Expect 3 channels after conversion to RGB
        (QImage.Format_RGB888, np.uint8, 3),  # Direct 3 channels
    ],
)
def test_qimage_to_numpy_array(image_format, expected_dtype, expected_channels):
    width, height = 100, 100  # Example dimensions
    qimage = create_test_qimage(width, height, image_format)
    np_array = qimage_to_numpy_array(qimage)

    assert np_array.dtype == expected_dtype, "Data type of numpy array should match"
    assert np_array.shape == (
        height,
        width,
        expected_channels,
    ), "Shape of the numpy array should be (height, width, channels)"
    # Check if the conversion retains the same color
    assert np.all(
        np_array == 128
    ), "The image array should have all pixels set to the grey value used"


def test_qimage_to_numpy_array_unsupported_format():
    width, height = 100, 100
    unsupported_format = QImage.Format_Invalid
    qimage = create_test_qimage(width, height, unsupported_format)
    with pytest.raises(ValueError):
        _ = qimage_to_numpy_array(qimage)


@pytest.mark.parametrize(
    "array, expected_exception, expected_message",
    [
        # Correct input, no exception expected
        (np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8), None, None),
        # Incorrect number of channels
        (
            np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8),
            AssertionError,
            "Expected 3 channels but got 4",
        ),
        # Incorrect dtype
        (
            np.random.rand(100, 100, 3).astype(np.float32),
            AssertionError,
            "Expected dtype np.uint8 but got float32",
        ),
        # Incorrect dimensions
        (
            np.random.randint(0, 256, (100, 100), dtype=np.uint8),
            ValueError,
            "Input array must have 3 dimensions.",
        ),
    ],
)
def test_numpy_array_to_qimage_varied_cases(
    array, expected_exception, expected_message
):
    if expected_exception:
        with pytest.raises(expected_exception) as e:
            qimage = numpy_array_to_qimage(array)
        assert expected_message in str(
            e.value
        ), f"Should raise {expected_exception} with specific message"
    else:
        qimage = numpy_array_to_qimage(array)
        # Check that the QImage was created with the correct properties
        assert (
            qimage.width() == array.shape[1] and qimage.height() == array.shape[0]
        ), "Dimensions of the QImage should match the array"
        assert qimage.format() == QImage.Format_RGB888, "Image format should be RGB888"
