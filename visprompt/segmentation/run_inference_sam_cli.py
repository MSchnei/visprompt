from functools import lru_cache
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


class SAMInference:
    def __init__(
        self,
        model_id="facebook/sam-vit-large",
        device="cpu",
    ):
        self.model_id = model_id
        self.device = device
        self.load_processor_and_model()

    @lru_cache(maxsize=None)
    def load_processor_and_model(self):
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.model = SamModel.from_pretrained("facebook/sam-vit-large").to(
            self.device
        )

    def run_inference(
        self,
        prompt_image: Image.Image,
        input_points: np.array,
    ) -> torch.Tensor:
        inputs = self.processor(
            prompt_image, input_points=input_points, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )


@click.command()
@click.option(
    "--model-id",
    type=click.STRING,
    default="facebook/sam-vit-large",
    help="Model ID on huggingface, optional",
)
@click.option(
    "--prompt-image",
    type=click.STRING,
    default="visprompt/examples/image_01.pg",
    help="Image for which we want to find segmentation mask",
)
@click.option(
    "-p",
    "--input-points",
    default=[100, 150],
    multiple=True,
    help="Point prompt for the segmentation task",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default="cpu",
    help="Device type to which model and model inputs will be allocated",
)
@click.option(
    "--output_dir",
    type=click.STRING,
    default="output_dir",
    help="Directory where result of the segmentation task will be saved",
)
def run_inference_sam_cli(
    model_id: str,
    prompt_image: str,
    input_points: int,
    device: str,
    output_dir: str,
):
    """CLI for running inference with SAM."""

    # process image
    image = Image.open(prompt_image).convert("RGB")
    input_points = [[list(input_points)]]

    inference_instance = SAMInference(model_id, device)
    mask = inference_instance.run_inference(
        prompt_image=image,
        input_points=input_points,
    )[0]

    output = Image.fromarray(
        (
            np.array(image)
            * (0.6 * mask.squeeze().numpy()[0][:, :, np.newaxis] + 0.4)
        ).astype(np.uint8)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / (
        "output_" + Path(prompt_image).stem + ".png"
    )
    output.save(out_path)


if __name__ == "__main__":
    run_inference_sam_cli()
