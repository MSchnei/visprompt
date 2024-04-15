from typing import List

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QPixmap
from tqdm import tqdm

from visprompt.sam_inference.run_inference_sam_cli import SAMInference
from visprompt.seggpt_inference.run_inference_seggpt_cli import SegGPTInference


def transform_points(points, scale_factor, x_offset, y_offset):
    transformed = []
    for point in points:
        x = (point.x() - x_offset) / scale_factor
        y = (point.y() - y_offset) / scale_factor
        transformed.append(QPoint(x, y))
    return transformed


def qimage_to_numpy_array(qimage):
    # Get image dimensions
    width = qimage.width()
    height = qimage.height()
    bytes_per_line = qimage.bytesPerLine()

    # Get pointer to the image data
    ptr = qimage.bits()

    # Convert to bytes (this step is important)
    bytearr = bytes(ptr)

    # Depending on the format of the QImage, process the byte data differently
    if qimage.format() == QImage.Format_RGB32:
        # Calculate expected bytes for a single row
        expected_bytes_per_line = width * 4
        if expected_bytes_per_line == bytes_per_line:
            arr = np.frombuffer(bytearr, np.uint8).reshape((height, width, 4))
        else:
            arr = (
                np.frombuffer(bytearr, np.uint8)
                .reshape(height, bytes_per_line)[:, : width * 4]
                .reshape(height, width, 4)
            )

        # Convert from RGBA to RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    elif qimage.format() == QImage.Format_RGB888:
        expected_bytes_per_line = width * 3
        if expected_bytes_per_line == bytes_per_line:
            arr = np.frombuffer(bytearr, np.uint8).reshape((height, width, 3))
        else:
            arr = (
                np.frombuffer(bytearr, np.uint8)
                .reshape(height, bytes_per_line)[:, : width * 3]
                .reshape(height, width, 3)
            )

    else:
        raise ValueError(f"Unsupported QImage format: {qimage.format()}")

    return arr


def numpy_array_to_qimage(array):
    # Ensure the array is shaped correctly
    if len(array.shape) == 3:
        assert array.shape[2] == 3, f"Expected 3 channels but got {array.shape[2]}"
        assert array.dtype == np.uint8, f"Expected dtype np.uint8 but got {array.dtype}"
        height, width, _ = array.shape
        format = QImage.Format_RGB888
    else:
        raise ValueError("Input array must have 3 dimensions.")

    # Create QImage from the NumPy array
    image = QImage(array.data, width, height, array.strides[0], format)

    # Make a copy to ensure the data is not freed
    image = image.copy()

    return image


def get_segmentation_image_from_sam(prompt_images: List[QImage], drawing_points_list):
    assert len(prompt_images) == len(
        drawing_points_list
    ), "Number of prompt images must match number of drawing points sets."

    segmentation_results = []
    print("Running segmentation with SAM...")

    inference_instance = SAMInference(device="cpu")

    for prompt_image, drawing_points in tqdm(zip(prompt_images, drawing_points_list)):
        prompt_image_np = qimage_to_numpy_array(prompt_image.toImage())
        # nb_images, nb_predictions, nb_points_per_mask, 2
        input_points = [[[[point.x(), point.y()] for point in drawing_points]]]

        mask = inference_instance.run_inference(
            prompt_image=prompt_image_np,
            input_points=input_points,
        )[0]

        # Modify the mask values to match QImage's expectations.
        mask = np.repeat(mask.squeeze().numpy()[0][:, :, np.newaxis], 3, axis=2)
        mask = mask.astype(np.uint8)
        mask[mask == 1] = 255

        segmentation_results.append(numpy_array_to_qimage(mask))

    return segmentation_results


def get_segmentation_image_from_seggpt(
    prompt_images: List[QPixmap],
    prompt_targets: List[QImage],
    user_images: List[QImage],
):
    print("Running segmentation with SegGPT...")

    prompt_images = [
        Image.fromarray(qimage_to_numpy_array(prompt_image.toImage()))
        for prompt_image in prompt_images
    ]
    prompt_targets = [
        Image.fromarray(qimage_to_numpy_array(prompt_target))
        for prompt_target in prompt_targets
    ]

    inference_instance = SegGPTInference(device="cpu")
    masks = []
    for user_image in tqdm(user_images):
        user_image = Image.fromarray(qimage_to_numpy_array(user_image.toImage()))
        mask = inference_instance.run_inference(
            input_image=user_image,
            prompt_images=prompt_images,
            prompt_targets=prompt_targets,
        )

        # Modify the mask values to match QImage's expectations.
        mask = np.repeat(mask.numpy()[:, :, np.newaxis], 3, axis=2)
        mask = mask.astype(np.uint8)
        mask[mask == 1] = 255
        masks.append(numpy_array_to_qimage(mask))

    return masks
