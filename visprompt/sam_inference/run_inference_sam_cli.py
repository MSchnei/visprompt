import click
import cv2
import numpy as np

from functools import lru_cache
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor


class SAMInference:
    def __init__(
        self, ckpt_path="models/sam_vit_l_0b3195.pth", model_type="vit_l", device="mps"
    ):
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.device = device
        self.model = None

    @lru_cache(maxsize=None)
    def _load_model(self):
        if self.model is None:
            print("Loading the SAM model ...")
            self.model = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
            self.model.to(device=self.device)

    def run_inference(
        self,
        prompt_image: np.array,
        input_point: np.array,
        input_label: np.array,
        multimask_output: bool,
    ):
        self._load_model()
        predictor = SamPredictor(self.model)
        predictor.set_image(prompt_image)
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
        )
        return masks.astype(np.uint8)


@click.command()
@click.option(
    "--ckpt-path",
    type=click.STRING,
    default="models/sam_vit_l_0b3195.pth",
    help="Path to downloaded model weights",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="vit_l",
    help="Name of the model",
)
@click.option(
    "--prompt-image",
    type=click.STRING,
    default="examples/hmbb_1.jpg",
    help="Image on which we perform SAM segmentation",
)
@click.option(
    "--input-point",
    type=click.INT,
    default=[100, 150],
    multiple=True,
    help="Point prompt for the segmentation task",
)
@click.option(
    "--input-label",
    type=click.INT,
    default=1,
    help="Corresponding label for input point. 1 means foreground / 0 means background",
)
@click.option("--multimask-output/--no-multimask-output", type=click.BOOL, default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default="cpu")
@click.option(
    "--output_dir",
    type=click.STRING,
    default="output_dir",
)
def run_inference_sam_cli(
    ckpt_path: str,
    model_type: str,
    prompt_image: str,
    input_point: int,
    input_label: int,
    multimask_output: bool,
    device: str,
    output_dir: str,
):
    """CLI for running inference with SAM."""

    # process image
    image = cv2.imread(prompt_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process input points and label
    input_point = np.array([input_point])
    input_label = np.array([input_label])

    inference_instance = SAMInference(ckpt_path, model_type, device)
    masks = inference_instance.run_inference(
        prompt_image=image,
        input_point=input_point,
        input_label=input_label,
        multimask_output=multimask_output,
    )

    # save
    for index in range(masks.shape[0]):
        output_mask = np.repeat(masks[index, ...][:, :, np.newaxis], 3, axis=2)
        output = Image.fromarray((image * (0.6 * output_mask + 0.4)).astype(np.uint8))
        out_path = Path(output_dir) / (
            "output_" + Path(prompt_image).stem + f"_{index}.png"
        )
        output.save(out_path)


if __name__ == "__main__":
    run_inference_sam_cli()
