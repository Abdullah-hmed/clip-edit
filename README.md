A GUI application that leverages CLIP and StyleGAN2 models to generate and semantically edit images using natural language prompts.

## Installation

Clone the repository:

```bash
git clone --recurse-submodules https://github.com/Abdullah-hmed/clip-edit.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Install PyTorch:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Install CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Place StyleGAN2 model weights (`.pkl` files) in the appropriate directory. Supported models include FFHQ, CelebA, and Metfaces. CelebA and FFHQ generally provide the best results for controlled edits.

## Downloading Models

Download compatible StyleGAN2 model weights from the [official repository](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/).

## Running the Application

Double-click `run.bat`, select your model file, wait for CLIP weights to download, and start editing!