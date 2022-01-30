# GraphCON
This repository contains the implementation to reproduce the numerical experiments 
of the preprint **Graph-Coupled Oscillator Networks**

<p align="center">
<img align="middle" src="./imgs/graphCON_figure.pdf" width="500" />
</p>

### Requirements
Main dependencies (with python >= 3.7):<br />
torch==1.9.0<br />
torch-cluster==1.5.9<br />
torch-geometric==2.0.3<br />
torch-scatter==2.0.9<br />
torch-sparse==0.6.12<br />
torch-spline-conv==1.2.1<br />
torchdiffeq==0.2.2

Commands to install all the dependencies in a new conda environment <br />
*(python 3.7 and cuda 10.2 -- for other cuda versions change accordingly)*
```
conda create --name graphCON python=3.7
conda activate graphCON

pip install ogb pykeops
pip install torch==1.9.0
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric
pip install wandb
```
