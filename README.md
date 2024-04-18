# visprompt
Welcome to *visprompt*, a repository and basic GUI for experimenting with visual prompting based on SAM and segGPT.

If you would like to learn more about visual prompting, please check out [the website accompanying this project](https://mschnei.github.io/visprompt/). 

Current version: 0.1.0

https://github.com/MSchnei/visprompt/assets/15090072/3243cd79-7373-48f3-a45d-b0caf8b7e6c0

## Installation

First, clone the project by running:
```bash
cd /home/folder/git/
git clone https://github.com/MSchnei/visprompt.git
```

Then set up a poetry environment by running:
```bash 
cd /home/folder/git/visprompt/
poetry shell
poetry install
```

## How to use

There are two modes in which you can use this package:
1. run and visualise segmentations via the GUI 
2. run sam and segGPT segmentation via commmand line

### GUI
To start the GUI, run:
```bash
poetry run python visprompt/gui/image_drawer.py
```

Then
- drop one or several image(s) for sam segmentation in the top-left window and draw a prompt per image
- drop one or several image(s) for segGPT segmentation in the bottom-left window panel
- click the `Submit` button

Running the application for the first time might take a while, since we need to download the models for huggingface hub.


### Command line
for segmentation with SAM, run:
```bash
poetry run python visprompt/sam_inference/run_inference_sam_cli.py --prompt-image /path/to/prompt_image.png -p 100 - p 150
```

for sementation with segGPT, run:
```bash
poetry run python visprompt/seggpt_inference/run_inference_seggpt_cli.py --input-image /path/to/input_image.png --prompt-images /path/to/prompt_image.png --prompt-targets /path/to/prompt_targets.png 
```
