Official PyTorch implementation of our TGRS paper: Deep Adaptive Pansharpening via Uncertainty-aware Image Fusion.

-------------------------------------------------
**Framework**

*UAPN*

<img src="https://github.com/keviner1/imgs/blob/main/UAPN.png?raw=true" width="600px">

*UAConv*

<img src="https://github.com/keviner1/imgs/blob/main/UAConv.png?raw=true" width="600px">

-------------------------------------------------
**Results**

*model complexity*
<img src="https://github.com/keviner1/imgs/blob/main/UAPN-complexity.png?raw=true">

*uncertainty estimation*
<img src="https://github.com/keviner1/imgs/blob/main/UAPN-uncertaintys.png?raw=true">

*comparison*
<img src="https://github.com/keviner1/imgs/blob/main/UAPN-comp.png?raw=true">

-------------------------------------------------
**We provide the training script as follows:**

-------------------------------------------------
**Dependencies**
* Python 3.8
* PyTorch 1.10.0+cu113

-------------------------------------------------
**Train**

Edit the data path in config files

> UAPN-B

python train.py --config 1

> UAPN-S

python train.py --config 2
