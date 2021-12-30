[![Paper](https://img.shields.io/badge/paper-arXiv%3A2103.01878-B31B1B.svg)](https://arxiv.org/abs/2112.12654)

# Quantum dynamics simulations beyond the coherence time on NISQ hardware by variational Trotter compression

[Noah F. Berthusen](https://noahberthusen.github.io), Thaís V. Trevisan, Thomas Iadecola, [Peter P. Orth](https://faculty.sites.iastate.edu/porth/)

### Abstract
We demonstrate a post-quench dynamics simulation of a Heisenberg model on present-day IBM quantum hardware that extends beyond the coherence time of the device. This is achieved using a hybrid quantum-classical algorithm that propagates a state using Trotter evolution and then performs a classical optimization that effectively compresses the time-evolved state into a variational form. When iterated, this procedure enables simulations to arbitrary times with an error controlled by the compression fidelity and a fixed Trotter step size. We show how to measure the required cost function, the overlap between the time-evolved and variational states, on present-day hardware, making use of several error mitigation methods. In addition to carrying out simulations on real hardware, we investigate the performance and scaling behavior of the algorithm with noiseless and noisy classical simulations. We find the main bottleneck in going to larger system sizes to be the difficulty of carrying out the optimization of the noisy cost function.

### Description
This repository includes information, code, scripts, and data to generate the figures in the paper.

### Requirements
* [optimparallel](https://pypi.org/project/optimparallel/)
* [cma](https://github.com/CMA-ES/pycma)
* [qiskit](https://github.com/Qiskit)
* [mitiq](https://github.com/unitaryfund/mitiq)

### Figures
All the codes used to create the figures in the paper are found in the **figure_scripts** folder. They are all written in Python, and use the matplotlib library.

### Data Generation
The main files to perform the algorithm detailed in the paper are described below. Generated data can be found in the **results** folder. The following files were designed to be ran on a computing cluster, and they may need to be modified to run on other systems. 
* ```Heisenberg.py``` Calculates the minimal infidelity for a given system size $M$ and ansatz depth $\ell$. Used to generated the data in Fig. 6 in the paper.
* ```Heisenberg-time.py``` Used to generate the data in Fig. 5(a)
* ```VTC.py``` Ideal implementation of the VTC algorithm for a given system size $M$, ansatz depth $\ell$, final time $dt$, and Trotter step size $\tau$. Used to generate the data for Fig. 7.
* ```circuitVTC.py``` Implementation of the VTC algorithm using circuits in Qiskit. In addition to the parameters above, the number of samples must be specified. Noiseless simulation as well as simulation with a noise model is possible. Used to generate data in Fig. 8 and Fig. 9(a).
* ```deviceCircuitVTC.py``` Identical implementation to ```circuitVTC.py```, just using a real IBM quantum backend. Used to get data for Fig. 9(b).

### Support
This material is based upon work supported by the National Science Foundation under Grant No. 2038010.

<img width="100px" src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png">