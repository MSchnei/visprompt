# visprompt
Welcome to *visprompt*, a repository and basic GUI for experimenting with visual prompting based on SAM and segGPT.

If you would like to learn more about visual prompting, please check out [the website accompanying this project](https://mschnei.github.io/visprompt/). 

Current version: 0.1.0

https://github.com/MSchnei/visprompt/assets/15090072/3243cd79-7373-48f3-a45d-b0caf8b7e6c0

## Installation

There are two ways in which you can install this repository:
1. standalone
2. add as a dependency in your poetry project

### Standalone
First, clone the project by running:
```shell
cd /home/folder/git/
git clone https://github.com/MSchnei/visprompt.git
```

Then set up a poetry environment by running:
```shell 
cd /home/folder/git/visprompt/
poetry shell
poetry install
```

### As a dependency
To add visprompt as a dependency to your poetry project, simply run:
```bash
poetry add visprompt
```

## How to use

There are two modes in which you can use visprompt:
1. run and visualise segmentations via the GUI 
2. run SAM and segGPT segmentation via commmand line

### Segmentation via GUI
To start the GUI from your terminal, run:
```shell
poetry run task gui
```

Alternatively, to start the GUI from a python shell, run:
```python
from visprompt import run_gui

run_gui()
```

Once the GUI opens
- drop one or several image(s) for sam segmentation in the top-left window and draw a prompt per image
- drop one or several image(s) for segGPT segmentation in the bottom-left window panel
- click the `Submit` button

Running the application for the first time might take a while, since we need to download the models from the huggingface hub.


### Segmentation via CLI
To run a SAM segmentation from your terminal, run the following:
```shell
poetry run task inference_sam --prompt-image /path/to/prompt_image.png -p 100 - p 150
```

To run a segGPT segmentation from your terminal, run
```shell
poetry run task inference_seggpt --input-image /path/to/input_image.png --prompt-images /path/to/prompt_image.png --prompt-targets /path/to/prompt_targets.png 
```

Alternatively, run the below from a python shell:
```python
from PIL import Image
from visprompt import SAMInference, SegGPTInference

# Set prompt_image and input_points for SAM segmentation
prompt_image = Image.open("/path/to/prompt_image.png").convert("RGB")
input_points = [[[100, 150]]]

# Run SAM segmentation
inference_instance = SAMInference()
mask = inference_instance.run_inference(
    prompt_image=prompt_image,
    input_points=input_points,
)

# Set input_image, prompt_images and prompt_targets for SegGPT segmentation
input_image = Image.open("/path/to/input_image.png").convert("RGB")
prompt_images = [Image.open("/path/to/prompt_image.png").convert("RGB")]
prompt_targets = [Image.open("/path/to/prompt_target.png").convert("RGB")]

# Run SegGPT segmentation
inference_instance = SegGPTInference(num_labels=1)
mask = inference_instance.run_inference(
    input_image=input_image,
    prompt_images=prompt_images,
    prompt_targets=prompt_targets,
)
```

## Contributing

Contributions are welcome! Before submitting a PR, please run:

```shell
make style
```

This will run `black`, `isort` and `flake8` on the code.

Unit tests can be executed via

```shell
make test
```