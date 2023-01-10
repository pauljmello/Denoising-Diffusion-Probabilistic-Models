Denoising Diffusion Probabilistic Models (DDPM)
======
An implementation of DDPM in PyTorch.
This unofficial pytorch implementation is based on the paper [Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Ho et al.
We provide the following coverage:
- Training on MNIST and N-dimensional Gaussian data.

## Example Diffusion Process
We train our models on MNIST and Gaussian datasets.
Below we demonstrate the forward diffusion process which sequentially adds noise to an image, leading to a gradual corruption.
![Forward Diffusion t=0 (image) -> t=1000 (noise)](Images/Example%20Gradual%20Corruption.jpg)

Below we demonstrate the reverse diffusion process which sequentially removes noise from an image, leading to a gradual restoration.
![Backward Diffusion t=1000 (noise) -> t=0 (image)](Images/Sample%20Plot%20Series/mnist/E%208%20T%200.jpg)

## Requirements
Download the requiremenets.txt file which contains the required packages including:
- PyTorch
- numpy
- matplotlib

## Citations & Acknowledgements
* [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
* [MI Neural Estimation](https://github.com/gtegner/mine-pytorch)

