import sys

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visprompt.gui.gui_image_utils import (
    get_segmentation_image_from_sam,
    get_segmentation_image_from_seggpt,
    transform_points,
)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


class ImageDisplay(QLabel):
    image_dropped = Signal()  # Create a new Signal

    def __init__(
        self,
        parent=None,
        allow_drops=False,
        allow_drawing=False,
        background_text=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QLabel.Box)
        self.image_list = []
        self.points = [[] for _ in range(len(self.image_list))]
        self.current_index = -1
        self.drawing = False
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.setAcceptDrops(allow_drops)
        self.allow_drawing = allow_drawing
        self.background_text = background_text
        # List to store original sizes of input images
        self.original_sizes = []

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        local_path = e.mimeData().urls()[0].toLocalFile()
        pixmap = QPixmap(local_path)
        if pixmap.isNull():
            print("Failed to load image.")
        else:
            self.original_sizes.append(pixmap.size())
            pixmap = pixmap.scaled(
                self.width(), self.height(), Qt.KeepAspectRatio
            )
            self.image_list.append(pixmap)
            self.current_index = len(self.image_list) - 1
            self.points.append([])
            self.setPixmap(pixmap)
            self.drawing = False
        self.update()
        self.image_dropped.emit()  # Emit the signal after the image is dropped

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.allow_drawing:
            self.drawing = True

    def mouseMoveEvent(self, e):
        if self.drawing and self.allow_drawing:
            self.points[self.current_index].append(e.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, e):
        if self.allow_drawing:
            self.drawing = False

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.background_text and (
            not self.pixmap() or self.pixmap().isNull()
        ):
            painter = QPainter(self)
            painter.setPen(Qt.gray)
            painter.setFont(QFont("Arial", 20))
            painter.drawText(self.rect(), Qt.AlignCenter, self.background_text)
        elif self.pixmap():
            pixmap = QPixmap(
                self.pixmap()
            )  # Make a copy of the current pixmap
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))

            pixmap_width, pixmap_height = pixmap.width(), pixmap.height()
            label_width, label_height = self.width(), self.height()

            # Calculate aspect ratios
            aspect_pixmap = pixmap_width / pixmap_height
            aspect_label = label_width / label_height

            # Calculate scale factors and offsets based on aspect ratio
            if aspect_pixmap > aspect_label:
                self.scale_factor = label_width / pixmap_width
                self.x_offset = 0
                self.y_offset = (
                    label_height - (pixmap_height * self.scale_factor)
                ) / 2
            else:
                self.scale_factor = label_height / pixmap_height
                self.x_offset = (
                    label_width - (pixmap_width * self.scale_factor)
                ) / 2
                self.y_offset = 0

            # Check for valid index before attempting to retrieve points
            if 0 <= self.current_index < len(self.points):
                current_points = self.points[self.current_index]

                for point in current_points:
                    # Apply scaling and offset
                    scaled_point = QPoint(
                        (point.x() - self.x_offset) / self.scale_factor,
                        (point.y() - self.y_offset) / self.scale_factor,
                    )
                    painter.drawPoint(scaled_point)

            painter.end()
            self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation Tool")

        central_widget = QWidget()
        main_layout = QVBoxLayout()

        self.prompt_display = ImageDisplay(
            allow_drops=True,
            allow_drawing=True,
            background_text="Drag prompt images here.",
        )
        self.segment_display = ImageDisplay()
        self.user_img_display = ImageDisplay(
            allow_drops=True, background_text="Drag test images here."
        )
        self.result_display = ImageDisplay()

        row1 = QHBoxLayout()
        row1.addWidget(self.prompt_display)
        row1.addWidget(self.segment_display)
        row2 = QHBoxLayout()
        row2.addWidget(self.user_img_display)
        row2.addWidget(self.result_display)
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)

        self.prev_prompt_button = QPushButton("Previous (Prompt Image)")
        self.next_prompt_button = QPushButton("Next (Prompt Image)")
        self.prev_prompt_button.clicked.connect(self.show_previous_prompt)
        self.next_prompt_button.clicked.connect(self.show_next_prompt)
        self.prev_prompt_button.setEnabled(False)
        self.next_prompt_button.setEnabled(False)

        self.prev_button = QPushButton("Previous (Test Image)")
        self.next_button = QPushButton("Next (Test Image)")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        prompt_button_layout = QHBoxLayout()
        prompt_button_layout.addWidget(self.prev_prompt_button)
        prompt_button_layout.addWidget(self.next_prompt_button)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        main_layout.addLayout(button_layout)
        main_layout.insertLayout(1, prompt_button_layout)

        self.submit_button = QPushButton("Run Segmentation")
        self.submit_button.clicked.connect(self.process_images)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)

        control_buttons = QHBoxLayout()
        control_buttons.addWidget(self.submit_button)
        control_buttons.addWidget(self.clear_button)
        main_layout.addLayout(control_buttons)

        self.save_button = QPushButton("Save Segmented Test Image")
        self.save_button.clicked.connect(self.save_result_image)
        main_layout.addWidget(self.save_button)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        # Connect the new Signal to an appropriate slot function
        self.prompt_display.image_dropped.connect(
            self.update_prompt_navigation_buttons
        )
        self.user_img_display.image_dropped.connect(
            self.update_navigation_buttons
        )
        # Set initial window size
        self.resize(800, 800)
        # To store the segmented images
        self.segmented_prompt_images = []
        self.segmented_images = []

    def update_prompt_navigation_buttons(self):
        has_images = len(self.prompt_display.image_list) > 0
        self.prev_prompt_button.setEnabled(has_images)
        self.next_prompt_button.setEnabled(has_images)

    def update_navigation_buttons(self):
        has_images = len(self.user_img_display.image_list) > 0
        self.prev_button.setEnabled(has_images)
        self.next_button.setEnabled(has_images)

    def show_previous_prompt(self):
        if len(self.prompt_display.image_list) == 0:
            return
        self.prompt_display.current_index = (
            self.prompt_display.current_index - 1
        ) % len(self.prompt_display.image_list)
        self.prompt_display.setPixmap(
            self.prompt_display.image_list[self.prompt_display.current_index]
        )
        if self.segmented_prompt_images:  # If segmented images are available
            self.segment_display.setPixmap(
                QPixmap.fromImage(
                    self.segmented_prompt_images[
                        self.prompt_display.current_index
                    ]
                )
            )

    def show_next_prompt(self):
        if len(self.prompt_display.image_list) == 0:
            return
        self.prompt_display.current_index = (
            self.prompt_display.current_index + 1
        ) % len(self.prompt_display.image_list)
        self.prompt_display.setPixmap(
            self.prompt_display.image_list[self.prompt_display.current_index]
        )
        if self.segmented_prompt_images:  # If segmented images are available
            self.segment_display.setPixmap(
                QPixmap.fromImage(
                    self.segmented_prompt_images[
                        self.prompt_display.current_index
                    ]
                )
            )

    def show_previous(self):
        if len(self.user_img_display.image_list) == 0:
            return
        self.user_img_display.current_index = (
            self.user_img_display.current_index - 1
        ) % len(self.user_img_display.image_list)
        self.user_img_display.setPixmap(
            self.user_img_display.image_list[
                self.user_img_display.current_index
            ]
        )
        if self.segmented_images:  # If segmented images are available
            self.result_display.setPixmap(
                QPixmap.fromImage(
                    self.segmented_images[self.user_img_display.current_index]
                )
            )

    def show_next(self):
        if len(self.user_img_display.image_list) == 0:
            return
        self.user_img_display.current_index = (
            self.user_img_display.current_index + 1
        ) % len(self.user_img_display.image_list)
        self.user_img_display.setPixmap(
            self.user_img_display.image_list[
                self.user_img_display.current_index
            ]
        )
        if self.segmented_images:  # If segmented images are available
            self.result_display.setPixmap(
                QPixmap.fromImage(
                    self.segmented_images[self.user_img_display.current_index]
                )
            )

    def process_images(self):
        # Run SAM Segmentation
        if (
            self.prompt_display.pixmap()
            and len(self.prompt_display.points) > 0
            and all(len(item) > 0 for item in self.prompt_display.points)
        ):
            self.run_sam_segmentation()
        else:
            raise Exception(
                "SAM segmentation prerequisites not met."
                "Ensure prompt image and drawing points are provided."
            )

        # Run segGPT Segmentation
        if len(self.user_img_display.image_list) == 0:
            print("No user images provided. Skipping segGPT segmentation.")
        else:
            self.run_seggpt_segmentation()

    def run_sam_segmentation(self):
        # Ensure there's one set of drawing points for each prompt image
        assert len(self.prompt_display.image_list) == len(
            self.prompt_display.points
        ), "Each prompt image should have an associated set of drawing points."

        transformed_points_list = [
            transform_points(
                points,
                self.prompt_display.scale_factor,
                self.prompt_display.x_offset,
                self.prompt_display.y_offset,
            )
            for points in self.prompt_display.points
        ]

        # Get segmentation results for all prompt images and drawing points.
        self.segmented_prompt_images = get_segmentation_image_from_sam(
            self.prompt_display.image_list, transformed_points_list
        )

        # Display the last segmentation result in the segment_display.
        # You can modify this as per your requirement.
        self.segment_display.setPixmap(
            QPixmap.fromImage(
                self.segmented_prompt_images[self.prompt_display.current_index]
            )
        )

    def run_seggpt_segmentation(self):
        # seg_image should be generated from SAM and be required for segGPT
        if (
            len(self.segmented_prompt_images) < 1
            or not self.prompt_display.pixmap()
        ):
            print(
                "segGPT prerequisites not met."
                "Ensure SAM segmentation is done and a prompt image is loaded."
            )
            return

        self.segmented_images = get_segmentation_image_from_seggpt(
            self.prompt_display.image_list,
            self.segmented_prompt_images,
            self.user_img_display.image_list,
        )
        self.result_display.setPixmap(
            QPixmap.fromImage(
                self.segmented_images[self.user_img_display.current_index]
            )
        )

    def save_result_image(self):
        if (
            not self.result_display.pixmap()
            or self.result_display.pixmap().isNull()
        ):
            print("No image to save.")
            return

        current_index = self.user_img_display.current_index
        if current_index < 0 or current_index >= len(
            self.user_img_display.original_sizes
        ):
            print("Invalid image index or original size not available.")
            return

        original_size = self.user_img_display.original_sizes[current_index]
        # Use IgnoreAspectRatio to match the original dimensions
        scaled_pixmap = self.result_display.pixmap().scaled(
            original_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )

        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpeg);;All Files (*)",
            options=options,
        )
        if filePath:
            scaled_pixmap.save(filePath)
            print(f"Image saved to {filePath}.")

    def clear_all(self):
        # Clear data for prompt_display
        self.prompt_display.image_list.clear()
        self.prompt_display.points.clear()
        self.prompt_display.original_sizes.clear()
        self.prompt_display.setPixmap(QPixmap())
        self.prompt_display.current_index = -1

        # Clear data for user_img_display
        self.user_img_display.image_list.clear()
        self.user_img_display.points.clear()
        self.user_img_display.original_sizes.clear()
        self.user_img_display.setPixmap(QPixmap())
        self.user_img_display.current_index = -1

        # Clear data for result_display and segment_display if necessary
        self.result_display.setPixmap(QPixmap())
        self.result_display.current_index = -1
        self.segment_display.setPixmap(QPixmap())
        self.segment_display.current_index = -1

        # Clear lists that store processed or segmented images
        self.segmented_prompt_images.clear()
        self.segmented_images.clear()

        # Update UI elements, such as disabling buttons
        self.prev_prompt_button.setEnabled(False)
        self.next_prompt_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # Refresh navigation buttons status based on cleared data
        self.update_prompt_navigation_buttons()
        self.update_navigation_buttons()


if __name__ == "__main__":
    run_gui()
