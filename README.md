# Neural-Network Quantum State Tomography
Implementation of the reconstruction algorithm presented in ["Neural-Network Quantum State Tomography"][1]. The current version allows the reconstruction of a quantum state, given a set of measurementis in a basis where the wavefunction has constant sign. 

The code is written in C++11, using the linear algebra library [Eigen3][2]. The parameters for the training are read from the command line as follows:

* `-nv`: Number of visible units
* `-nh`: Number of hidden units
* `-w `: Width of initial weights distribution
* `-nc`: Numer of sampling chains
* `-cd`: Number of Gibbs updates in Contrastive Divergence
* `-lr`: Learning rate
* `-l2`: L2 regularization constant
* `-bs`: Batch size
* `-ns`: Numer of training samples
* `-ep`: Number of training iterations 

[1]: https://arxiv.org/abs/1703.05334 "nnqst"
[2]: https://eigen.tuxfamily.org

##### The code will undergo a series of updates in the near-future, incorporating additional algorithms and features.


