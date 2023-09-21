# visprompt
A simple GUI for experimenting with visual prompting (SAM + segGPT)

Current version: 0.1.0

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

We need to install detectron2 using pip because there is no build for macOS available on pypi:
```bash
poetry run python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Likewise, we need to install SAM:
```bash
poetry run python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

We also need to download the SAM and segGPT models:
```bash
mkdir models && cd models
# SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# segGPT model
wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth
```

## How to use

There are two modes in which you can use this package:
1. run and visualise segmentations via the GUI 
2. run sam and segGPT segmentation via commmand line

### GUI
To start the GUI, run:
```python
poetry run python visprompt/gui/image_drawer.py
```
Then
- drop one or several image(s) for sam sementation in the top-left window and draw a prompt per image
- drop one or several image(s) for segGPT sementation in the bottom-left window panel
- click the `Submit` button

### Command line
for segmentation with SAM, run:
```python
poetry run python visprompt/sam_inference/run_inference_sam_cli.py
```

for sementation with segGPT, run:
```python
poetry run python visprompt/seggpt_inference/run_inference_seggpt_cli.py
```