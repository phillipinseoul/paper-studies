# paper-studies
For storing insightful research papers that I have read recently!

#### 1. Local Deep Implicit Functions for 3D Shape [[arXiv]](https://arxiv.org/abs/1912.06126)
##### Key Concepts
* 3D Computer Vision
* 3D shape representation
* Surface reconstruction
* Space decomposition
* Deep implicit function (DIF)
* Structured implicit function (SIF)
* Latent vector

##### Summary
* The goal of this project is to learn a 3D shape representation that enables accurate surface reconstruction, compact storage, efficient computation, consistency for similar shapes, generalization across diverse shape categories, and inference from depth camera observations.
* Introduce Local Deep Implicit Functions (LDIF), a 3D shape representation that decomposes space into a structured set of learned implicit functions.
* Provide networks that infer the space decomposition and local deep implicit functions from a 3D mesh or posed depth image.
* During experiments, we find that it provides 10.3 points higher surface reconstruction accuracy (F-Score) than the state-of-the-art (OccNet), while requiring fewer than 1% of the network parameters.

##### Important Lines
> Most recently, deep implicit functions (DIF) have been shown to be highly effective for reconstruction of individual objects. ... This approach achieves state
of the art results for several 3D shape reconstruction tasks. ... But they support limited shape complexity, generality, and computational efficiency.

> Structured Implicit Functions (SIF) represents an implicit function as a mixture of local Gaussian functions. ... But they did not use these structured decompositions for accurate shape reconstruction due to the limited shape expressivity of their local implicit functions (Gaussians).

> LDIF latent vector is decomposed into parts associated with local regions of space (SIF Gaussians), which makes it more scalable, generalizable, and computationally efficient.

> We propose a new 3D shape representation, Local Deep Implicit Functions (LDIF). The LDIF is a function that can be used to classify whether a query point x is inside or outside a shape.

> LDIF replaces the (possibly long) single latent code of a typical DIF with the concatenation of N pairs of analytic parameters and short latent codes (i.e., the global implicit function is decomposed into the sum of N local implicit functions.)

> The input to the system is a 3D surface or depth image, and the output is a set of shape element parameters and latent codes for each of
N overlapping local regions.





