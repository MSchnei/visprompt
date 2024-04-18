import pytest
from PySide6.QtCore import QPoint

from visprompt.gui.gui_image_utils import (
    get_segmentation_image_from_sam,
    get_segmentation_image_from_seggpt,
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
