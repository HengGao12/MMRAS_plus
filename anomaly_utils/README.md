



## Installation

For Mask2former installation, please run the following code.

```sh
conda create -n ras python=3.7
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2 -f qhttps://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html

pip install opencv-python
pip install -r requirements.txt

# if the gcc version is too low, please run the following code
# conda install -c omgarcia gcc-6

cd mask2former/modeling/pixel_decoder/ops
pip install -e .

pip install easydict
pip install seaborn
pip install ood_metrics
pip install kornia
```

For LLM installation, 

## Get Started

