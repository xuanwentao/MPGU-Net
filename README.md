# MPGU-Net: Multi-Model Physics-Guided Network for Nonlinear Hyperspectral Unmixing
Xuanwen Tao, Bikram Koirala, Behnood Rasti, Antonio Plaza, Paul Scheunders

**1. Abstract**

Hyperspectral unmixing remains a challenging problem in complex scenes, where the linear model cannot describe the spectral reflectance of mixtures of pure materials. Although nonlinear mixing models have been proposed to address this challenge, existing methods are either supervised, i.e., require endmembers of the dataset as prior information, or are suitable only for datasets containing either pure pixels or near-pure pixels. When pure pixels are unavailable in the dataset, a dataset simplex has to be reconstructed in order to generate virtual endmembers. Unfortunately, the popular minimum simplex volume constraint (MSVC) often used for linear datasets cannot be directly applied to nonlinear datasets. To tackle this challenge, in this work, we propose a novel strategy called linearizing space transformation (LST) that estimates the linear component of a nonlinear dataset by analytically inverting the nonlinear dataset. MSVC can then be applied to this linear component to estimate the endmembers of the nonlinear dataset. LST  can be applied to nonlinear datasets that can be expressed as an explicit combination of linear and nonlinear components. The polynomial mixing models are a class of models that fulfill this criterion. In this work, we theoretically demonstrate the effectiveness of the proposed strategy for two popular mixing models from the literature, i.e., the polynomial post-nonlinear mixing model (PPNM) and the multilinear mixing model (MLM). In order to estimate endmembers, abundance maps, and nonlinear parameters of the mixing models, an end-to-end deep network called MPGU-Net is proposed. The network consists of three modules: (1) a variational autoencoder-based endmember generation network for robust spectral signature estimation; (2) an abundance estimation network enhanced by a convolutional block attention module for accurately estimating abundance maps; and (3) a nonlinearity parameter estimation network for estimating pixel-wise nonlinear interaction processes. In contrast to existing methods that model nonlinearity through a nonlinear activation function in the last layer of the network, our approach models nonlinearity explicitly through the nonlinear mixing equation. LST is not a direct component of the MPGU-Net framework, but is incorporated into the loss function, enabling more accurate estimation of endmembers and fractional abundances.

**2. Overview**

![2-MPGU](https://github.com/xuanwentao/MPGU-Net/blob/main/FrameworkV02.png)


**3. Citation**

Please kindly cite the paper if this code is useful and helpful for your research.

X. Tao, B. Koirala, B. Rasti, A. Plaza and P. Scheunders, "MPGU-Net: Multimodel Physics-Guided Network for Nonlinear Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 5518522-5518522, 2026, Art no. 5518522, doi: 10.1109/TGRS.2026.3705097.

     @article{tao2026mpgu,
      title={MPGU-Net: Multi-Model Physics-Guided Network for Nonlinear Hyperspectral Unmixing},
      author={Tao, Xuanwen and Koirala, Bikram and Rasti, Behnood and Plaza, Antonio and Scheunders, Paul},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2026},
      publisher={IEEE}
      }

**4. Contact Information**

Xuanwen Tao: txw_upc@126.com<br> 
