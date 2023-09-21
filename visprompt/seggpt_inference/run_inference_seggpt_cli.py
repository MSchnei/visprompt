import click
import torch
import numpy as np
from functools import lru_cache
from PIL import Image
from pathlib import Path
import visprompt.seggpt_inference.models_seggpt as models_seggpt
from visprompt.seggpt_inference.seggpt_engine import inference_image
from typing import List


class SegGPTInference:
    def __init__(
        self,
        ckpt_path="models/seggpt_vit_large.pth",
        model_type="seggpt_vit_large_patch16_input896x448",
        seg_type="instance",
        device="cpu",
    ):
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.seg_type = seg_type
        self.device = device
        self.model = None

    @lru_cache(maxsize=None)
    def _load_model(self):
        if self.model is None:
            print("Loading the SegGPT model...")
            self.model = getattr(models_seggpt, self.model_type)()
            self.model.seg_type = self.seg_type
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.model.to(device=self.device)
            self.model.eval()

    def run_inference(
        self,
        input_image: Image.Image,
        prompt_images: List[Image.Image],
        prompt_targets: List[Image.Image],
    ):
        self._load_model()
        mask = inference_image(
            self.model,
            self.device,
            input_image,
            prompt_images,
            prompt_targets,
        )
        return mask


@click.command()
@click.option(
    "--ckpt-path",
    type=click.STRING,
    default="models/seggpt_vit_large.pth",
    help="Path to downloaded model weights",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="seggpt_vit_large_patch16_input896x448",
    help="Name of the model",
)
@click.option(
    "--input-image",
    type=click.STRING,
    default="examples/hmbb_2.jpg",
    help="Image for which we want to find segmentation mask",
)
@click.option(
    "--prompt-images",
    type=click.STRING,
    multiple=True,
    default=["examples/hmbb_1.jpg"],
    help="Image on which we specify segmentation task",
)
@click.option(
    "--prompt-targets",
    type=click.STRING,
    multiple=True,
    default=["examples/hmbb_1_target.png"],
    help="Segmentation prompt for prompt image",
)
@click.option(
    "--seg-type",
    type=click.STRING,
    required=False,
    help="Image on which we specify segmentation task",
)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default="cpu")
@click.option(
    "--output_dir",
    type=click.STRING,
    default="output_dir",
)
def run_inference_seggpt_cli(
    ckpt_path: str,
    model_type: str,
    input_image: str,
    prompt_images: str,
    prompt_targets: str,
    seg_type: str,
    device: str,
    output_dir: str,
):
    """CLI for running inference with SegGPT."""

    # Process images
    image = Image.open(input_image).convert("RGB")
    prompt_images = [
        Image.open(prompt_image).convert("RGB") for prompt_image in prompt_images
    ]
    prompt_targets = [
        Image.open(prompt_target).convert("RGB") for prompt_target in prompt_targets
    ]

    inference_instance = SegGPTInference(ckpt_path, model_type, seg_type, device)
    mask = inference_instance.run_inference(
        input_image=image,
        prompt_images=prompt_images,
        prompt_targets=prompt_targets,
    )

    output = Image.fromarray(
        (np.array(image) * (0.6 * mask / 255 + 0.4)).astype(np.uint8)
    )
    out_path = Path(output_dir) / ("output_" + Path(input_image).stem + ".png")
    output.save(out_path)


if __name__ == "__main__":
    run_inference_seggpt_cli()
