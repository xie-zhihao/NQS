# NQS

## WELCOME! 

Clarification: 

For all the information below, they are all tentative! I am fully opened to any modification, idea or even brand-new plans! I DON’T assume any background (for this project QM, Abstract algebra, Computation complexity…… can be a plus, but they are not necessary), If you have any thoughts about this topic, interests or personal information, you are always welcomed to leave me a message! Long discussions/suggestions/demonstrations are also very welcomed.  

 

# Contents

- [Brief Introduction](#Brief-introduction)
- [Related Papers](#Related-Papers)
- [My Comments and thoughts](#My-Comments-and-thoughts)
- [Possible Realization structure](#Possible-Realization-structure)
- [More Personal Information](#More-Personal-Information)



Keywords: Symmetry-aware deep learning, stochastic optimization, Monte Carlo methods, quantum many-body basics

# Brief introduction

Here is some information of what Neural Quantum States (NQS) is.

In our math context, NQS corresponds to modeling **high-dimensional probability measures** with structured correlations, much like energy-based models or variational inference extended into quantum space. 

In physics context, where it originated, NQS is a method of approximating wavefunction by neural networks. This is aimed at solving quantum many-body problems by neural networks, and for solving quantum many-body problems, other famous approaches include density functional theory (DFT) and quantum Hamiltonian simulation. 

More technically, this is the realm of compressing high dimensional data into a lower-dimensional manifolds, because of quantum entanglements, the dimensions of Hilbert space of quantum states grows exponentially, and this method is introduced for lowering the dimensions while trying to keep vital information captured such that we can do easier simulation computations. 

# Related Papers

The following are few papers that you can take a look. Some files are in the repository, and links are also attached. 

[1] [Solving the quantum many-body problem with artificial neural networks | Science](https://www.science.org/doi/10.1126/science.aag2302)

This is the very first article that introduces the idea of neural quantum states in 2017. This also includes introduction to Transversed field Ising model and Heisenberg model which is commonly used as benchmarks for Quantum Many-body computation methods. 

[2]  [Neural-network quantum states for many-body physics | The European Physical Journal Plus](https://link.springer.com/article/10.1140/epjp/s13360-024-05311-y)

From 2024, this article mainly focuses on variational Monte Carlo (VMC) approaches. Benchmark systems and bottlenecks of computations are also described. 

[3] [From architectures to applications: a review of neural quantum states - IOPscience](https://iopscience.iop.org/article/10.1088/2058-9565/ad7168)

Also from 2024, this article gives very clear NQS architectures review. These architectures include Convolutional Neural Networks (CNNs) and Group Convolutional Neural Networks (GCNN). 

# My Comments and thoughts

**My comment, Why is NQS important and what we are working on. 

One important and inevitable topic is Ising model. This model is the simplest many-body model, also it is the father of Hopfield Neural Network, or in fact, this is the ancestor of modern artificial intelligence. Hopfield Neural network (Hopfield Model), the model that wins 2024 physics Nobel Prize, is a type of Ising model with fully-connected topology and a refined way of learning by updating weights, this is also where Boltzmann machine originated from. There are many settings of topology and traits which lead to different kinds of models. Besides AI, this is also the base of Spin Glass Models, which plays very important roles in computational physical sciences. 

NQS VMC is one of the rare approaches that can provide common norms for wavefunction computations. Some common methods might face sign problem due to rugged solution space, in optimization context, some common methods might become quasi local search algorithm due to characteristics of objective function.

One exciting news here, Chancellor of Zhejiang University(浙江大学), Ma Yanming(马琰明院士)’s group claimed the FIRST realization of Room-temperature superconductor in human history this week. They realized with Ternary La-Sc-H system under the conditions of temperature 298 K and pressure 130-150 GPa. The arxiv link is attached here [5]: [[2510.01273\] Room-Temperature Superconductivity at 298 K in Ternary La-Sc-H System at High-pressure Conditions](https://arxiv.org/abs/2510.01273)

Why am I addressing this issue particularly here? To be clear, NQS is not only tied to Physics, Chemistry, but also has potential or has applied to stock market prediction, quantum machine learning, optimization, information science and etc.. The superconductor example above is a perfect instance of importance and potential of NQS. According to the author, their realization experiment is based the prediction method [6] that based on “swarm-intelligence-based CALYPSO structure prediction method in combination with first-principles calculations.” This means after they used their self-contained CALYPSO structure prediction to derive their model’s geometries, they applied density functional theory (DFT) to calculate and predict. Common reviews state that DFT is good at weak correlated systems, and for strong correlated systems it often needs corrections and DFTs cannot capture loads of effects. This limitation of prediction also leads to the limitation of this astonishing achievement, though they did realize Room-temperature Superconductor, the pressure required is also skyrocketed high. In brief, DFTs are not able to compute high temperature superconductors in normal pressure, and approaches of effectively predict is to have breakthroughs in BCS or strong correlated systems, and NQS is firmly tied to strong correlated system. The link of the prediction paper [6] they used is also attached here: [Predicted hot superconductivity in LaSc2H24 under pressure | PNAS](https://www.pnas.org/doi/10.1073/pnas.2401840121)

# Possible Realization structure

Possible Realization Structure 

My outline follows Dr. Rajah’s paper [7] partially. This work by Dr. Rajah and other authors built a neural quantum state based on ConvNeXt [8], a very famous modern convolutional neural network architecture inspired Transformer. To be clear, reproduction of ConvNeXt quantum states is NOT the main theme here since this paper represents SOTA level in this field. I am just here trying to cite its structure.

First task is to build up translationally-symmetric neural quantum states.

The steps are:

1. Build a wavefunction that captures good approximation to the sign structure.
2. Use embedding layer to partition lattice into P patches. The embedding is written as follows:

$\tilde{\sigma}^\mu_r = f^s (\sigma)^\mu_r = \Sigma_x W^\mu_x \sigma_{r,x} $, 

Where \mu indices the embedding dimension (features) of embedding layer total dimension $N_f$, and $\sigma_{r,x}$ is the input configuration in the computational basis.

3. Encode embedded patches through a series of translationally-equivariant transformations, 

$\tilde{\sigma}_{\hat{t} r} [l+ 1] = g^e (t^{-1} \tilde{\sigma}_r [l])$,

where $g^e$ is the encoder function, t is an element of the translation group, T, l labels the layers of the encoder.

4. Make network translationally-symmetric by projecting to a specific momentum, $\mathbb{k}$,

$\Psi_\theta^{(k)} = \frac{1}{|T|} \sum_{t \in T} e^{-i \mathbb{k} \cdot \mathbb{r_t}} \Psi_\theta (t^{-1} \sigma)$, 

and this is a neural quantum state being constructed via symmetry projection, this is equivalent to group representation projection operator: $\Psi^{(k)} = \frac{1}{|T|} \sum_{t \in T} \chi_k^*(t)\, \hat{T}_t \Psi$. 

After building neural quantum states, we use the adjustable wavefunction $\psi_\theta (s)$ approximated neural network (from above) to implement variational Monte Carlo to approach ground state. Tentative steps are listed below:

1. Select a benchmark model, suppose we selected the (spin -1/2) Heisenberg model, the Hamiltonian is:

$H = J \sum_{\langle ij \rangle} \mathbb{S}_i \cdot \mathbb{S}_j$,

$\langle ij \rangle$ represents neighboring sites. The variational energy is: $E(\theta) = \frac{\langle\psi_\theta |H| \psi\rangle }{\langle \psi_\theta | \psi_\theta \rangle}$

2. Sampling, the distribution is quantum probability $|\psi_\theta (s)|^2$, and it is implemented through Metropolis-Hastings algorithm. Additional techniques including symmetry, sign structure, autoregression etc.. can also be discussed here. 
3. Updates such that we can search ground state:

Stochastic gradient type method, normal way includes quantum version SGD: stochastic reconfiguration, or we can consider Adam, AdamW (I am sorry, I do recognize ADAM.), improved optimizer for NQS MinSR[9] and a lot of other options. The updating formula is:

$\theta \gets \theta - \eta \nabla_\theta E$.

Here is an analogous visualization of how this search would be. The file [10] https://github.com/xie-zhihao/NQS/blob/main/ising2D_ZX.m is in the repository, you can run it by yourself. This is an Ising model simulated by metropolis algorithm. 

<figure class="image" style="width:50%">
  <img src="https://github.com/xie-zhihao/NQS/blob/main/ising2D_ZX.gif" alt="">
  <figcaption><i></i></figcaption>
</figure>

 

How do we know whether it is effective or not?
 Here is a simple Density Matrix Renormalization Group example that provides “Standards” for some models [11], this is written by Prof. Lin Lin @UC Berkeley. Reminder: This is written in Julia, if you are using Jupyter notebook, you have to connect notebook to Julia and use Julia kernel to execute. 

# More Personal Information

More about me:
 I am a visitor from Berkeley math, my focus is computing and Geometry. I am interested in quantum information, statistical mechanics and differential geometry for physics. I used to work as a Data engineer shortly in China’s largest private bank’s R&D department so I also have some familiarity with industrial realization of NLP CV tasks.  

Just reiterate, For all the information above, they are all tentative! I am fully opened to any modification, idea or even brand-new plans! I DON’T assume any background (for this project QM, Abstract algebra, Computation complexity…… can be a plus, but they are not necessary), If you have any thoughts about this project, interests or personal information, you are always welcomed to leave me message! Long discussion/suggestions/demonstrations are also very welcomed.  

Email: [xiezhihao-justin@berkeley.edu](mailto:xiezhihao-justin@berkeley.edu)

Tel/微信: +1 510 501 6079

Zhihao

2025/10/08
