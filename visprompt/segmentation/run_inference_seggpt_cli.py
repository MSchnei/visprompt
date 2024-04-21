from functools import lru_cache
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from PIL import Image
from transformers import SegGptForImageSegmentation, SegGptImageProcessor


class SegGPTInference:
    def __init__(
        self,
        model_id="BAAI/seggpt-vit-large",
        num_labels=1,
        device="cpu",
    ):
        self.model_id = model_id
        self.num_labels = num_labels
        self.device = device
        self.load_processor_and_model()

    @lru_cache(maxsize=None)
    def load_processor_and_model(self):
        self.processor = SegGptImageProcessor.from_pretrained(self.model_id)
        self.model = SegGptForImageSegmentation.from_pretrained(
            self.model_id
        ).to(self.device)

    def run_inference(
        self,
        input_image: Image.Image,
        prompt_images: List[Image.Image],
        prompt_targets: List[Image.Image],
    ) -> torch.Tensor:
        nb_prompts = len(prompt_images)
        inputs = self.processor(
            images=[input_image] * nb_prompts,
            prompt_images=prompt_images,
            prompt_masks=prompt_targets,
            num_labels=self.num_labels,
            return_tensors="pt",
            feature_ensemble=True if nb_prompts > 1 else False,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [input_image.size[::-1]] * nb_prompts
        mask = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes, num_labels=self.num_labels
        )[0]
        return mask


@click.command()
@click.option(
    "--model-id",
    type=click.STRING,
    default="BAAI/seggpt-vit-large",
    help="Model ID on huggingface, optional",
)
@click.option(
    "--input-image",
    type=click.STRING,
    default="visprompt/examples/hmbb_2.jpg",
    help="Image for which we want to find segmentation mask",
)
@click.option(
    "-pi",
    "--prompt-images",
    type=click.STRING,
    multiple=True,
    default=["visprompt/examples/hmbb_1.jpg"],
    help="Image on which we specify segmentation task",
)
@click.option(
    "-pt",
    "--prompt-targets",
    type=click.STRING,
    multiple=True,
    default=["visprompt/examples/hmbb_1_target.png"],
    help="Segmentation target for prompt image",
)
@click.option(
    "--num-labels",
    type=click.INT,
    default=1,
    help="Number of labels in the segmentation task",
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
def run_inference_seggpt_cli(
    model_id: str,
    input_image: str,
    prompt_images: str,
    prompt_targets: str,
    num_labels: int,
    device: str,
    output_dir: str,
):
    """CLI for running inference with SegGPT."""

    # Process images
    image = Image.open(input_image).convert("RGB")
    prompt_images = [
        Image.open(prompt_image).convert("RGB")
        for prompt_image in prompt_images
    ]
    prompt_targets = [
        Image.open(prompt_target).convert("RGB")
        for prompt_target in prompt_targets
    ]

    inference_instance = SegGPTInference(model_id, num_labels, device)
    mask = inference_instance.run_inference(
        input_image=image,
        prompt_images=prompt_images,
        prompt_targets=prompt_targets,
    )

    output = Image.fromarray(
        (
            np.array(image) * (0.6 * mask.numpy()[:, :, np.newaxis] + 0.4)
        ).astype(np.uint8)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / ("output_" + Path(input_image).stem + ".png")
    output.save(out_path)


if __name__ == "__main__":
    run_inference_seggpt_cli()
