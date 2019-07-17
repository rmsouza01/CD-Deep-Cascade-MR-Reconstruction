# CD-Deep-Cascade-MR-Reconstruction

This repository investigates the usage of Cascades of Convolutional Neural Networks (CNN) for Magnetic Resonance (MR) image reconstruction.

Most of the data used in this repository is publicly available as part of the [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home), which currently has an open [MR reconstruction challenge](https://sites.google.com/view/calgary-campinas-dataset/home/mr-reconstruction-challenge).

Preliminary work on single-channel MR reconstruction was published at the 2019 Medical Imaging with Deep Learning conference:

[Souza, Roberto, R. Marc Lebel, and Richard Frayne. "A Hybrid, Dual Domain, Cascade of Convolutional Neural Networks for Magnetic Resonance Image Reconstruction." Proceedings of Machine Learning Research–XXXX 1 (2019): 11.](http://proceedings.mlr.press/v102/souza19a/souza19a.pdf)


![Sample Multi-channel Reconstruction](./Figs/midl_mc_5x.gif)

In this repository we  will look into:

- Advantages of Data Consistency (DC) layers; 
- Cascades of flat un-rolled structures and cascades of U-nets;
- Cross domain cascades (k-space and image domains);
- Different sampling pattern distributions;
- Potential advantages of including adversarial components;


Updated: 17 July 2019






