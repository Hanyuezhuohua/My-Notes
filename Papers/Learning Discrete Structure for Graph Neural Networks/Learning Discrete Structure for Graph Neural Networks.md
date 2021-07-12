# Learning Discrete Structure for Graph Neural Networks

## Abstract

1. jointly learn the graph structure and the parameters of graph convolutional networks (GCNs) by approximately solving a program that learns a discrete probability distribution on the edges of the graph

2. apply GCNs not only in scenarios where the given graph is incomplete or corrupted but also in those where a graph is not available

## Introduction 

1. Relational learning, therefore, does not make the assumption of independence between data points but models their dependency explicitly. 

2. Graph neural networks are one such class of algorithms that are able to incorporate sparse and discrete dependency structures between data points.

3. Previous ways:

   * KNN: the efﬁcacy of the resulting models hinges on the choice of k and, more importantly, on the choice of a suitable similarity measure over the input features
   * kernel matrix: calculate the similarity, but introduce a dense dependency structure.

## Background

1. Graph Theory Basics:

   * A graph $G$ is a pair $(V, E)$
   * $L= D -A$ 
   * $\mathcal{H}_{N}$ is the set of all $N \times N$ adjacency matrices

2. Graph Neural Networks

   * a feature matrix $X \in \mathcal{X}_N \subset \mathbb{R}^{N \times n}$

   * a class labels $\mathcal{Y}$ and a labeling function $y: V \rightarrow \mathcal{Y}$

   * a set of training nodes $V_{\text{Train}}$

   * a function $f_w:\mathcal{X}_N \times \mathcal{H}_N \rightarrow \mathcal{Y}^{N}$

   * a regularized empirical loss $L(w,A)=\sum_{v\in V_{\text{Train}}}l(f_w(X,A)_v,y_v)+\Omega(w)$ 

3. Bilevel Programming in Machine Learning

   * Two objective functions $F$ and $L$ 
   
   * Two sets of variables $\theta \in \mathbb{R}^m$ and $w \in \mathbb{R}^d$
   
   * $\underset{\theta, w_{\theta}} \min F(w_\theta,\theta)$ such that $w_\theta \in \arg \underset{w} \min L(w, \theta)$
   
   * solve above-mentioned problem: the minimization of $L$ $\Rightarrow$ the repeated application of an iterative optimization dynamics $\Phi$ such as (stochastic) gradient descent
   
     Let $w_{\theta, T}$ denote the inner variables after $T$ iterations of dynamics $\Phi$, $w_{\theta,T}=\Phi(w_{\theta,T-1},\theta)=\Phi(\Phi(w_{\theta,T-2},\theta),\theta)$ 
   
     $\frac{\partial F(w_{\theta,T},\theta)}{\partial w}=\partial_wF(w_{\theta,T},\theta)\nabla_\theta w_{\theta,T}+\partial_\theta F(w_{\theta,T},\theta)$

## Learning Discrete Graph Structures

* Framing it as a bilevel programming problem whose outer variables are the parameters of a generative probabilistic model for graphs.

* Based on truncated reverse-mode algorithmic differentiation & hypergradient estimation

1. Jointly Learning the Structure and Parameters

   * from:
     $$
     \underset{A}\min F(w_A, A) = \sum_{v \in V_{val}}l(f_{w_A}(X,A)_v,y_v) \text{ such that } w_{A} \in \arg \underset{w}\min L(w, A)
     $$

   * to: $\hat{\mathcal{H}}_N=Conv(\mathcal{H}_N) + \theta \in \hat{\mathcal{H}}_N + \mathcal{H}_N \ni A \text{~} Ber(\theta)$
     $$
     \underset{\theta \in \hat{H}_N}\min \mathbb{E}_{A\text{~}Ber(\theta)}[F(w_\theta, A)] \text{ such that } w_{\theta} \in \arg \underset{w}\min  \mathbb{E}_{A\text{~}Ber(\theta)}[L(w, A)]
     $$
     
*  Only be able to find approximate stochastic solutions
   
* $f_w^{exp}(X)=\mathbb{E}_A[f_w(X,A)] = \sum_{A \in \mathcal{H}_N}p_\theta(A)f_w(X,A)$
  
  an empirical estimate of the output as $\hat{f}_w(X)=\frac{1}{S}\sum_{i=1}^Sf_w(X,A_i)$
  
* sample a new graph: $O(N^2)$ 
  
* compute $\hat{f}_w: O(SCd)$  where $C= \sum_{ij}\theta_{ij} $ is the expected numbers of edges and $d$ is the dimension of the weights
  
2. Structure Learning via Hypergradient Descent

   * Inner problem: $\mathbb{E}_{A\text{~}Ber(\theta)}[L(w, A)] = \sum_{A\in\mathcal{H}_N}P_\theta(A)L(w,A)$

   * to: $w_{\theta,t+1}=\Phi(w_{\theta,t},A_t)=w_{\theta,t}-\gamma_t\nabla L(w_{\theta,t},A_t)$

   * Estimating $\nabla_\theta \mathbb{E}_{z\text{~}P_\theta}[h(z)]:$

     * a differentiable and reversible sampling path $sp(\theta, \epsilon)$ for $P_\theta$ with $z = sp(\theta, \epsilon)$ for $\epsilon \text{~} P_{\epsilon}$
     * $\nabla_\theta\mathbb{E}_{z\text{~}P_\theta}[h(z)]=E_{\epsilon\text{~}P_\epsilon}[\nabla_\theta h(sp(\theta,\epsilon))] = \mathbb{E}_{z\text{~}P_\theta}[\nabla_zh(z)\nabla_\theta z]$

   * $\nabla_\theta\mathbb{E}_{A\text{~}Ber(\theta)}[F(w_{\theta,T},A)]\approx\mathbb{E}_{A\text{~}Ber(\theta)}[\nabla_AF(w_{\theta,T},A)] = \mathbb{E}_{A\text{~}Ber(\theta)}[\partial_wF(w_{\theta,T},A)\nabla_Aw_{\theta,T}+\partial_AF(w_{\theta,T},A)]$

   * Inner: decrease condition

     Outer: early stopping criterion by validation set

## Experiments

Three main objectives:

* Evaluated LDS on node classification problems where a graph structure is avaliable but where a certain fraction of edges is missing.
* Evaluated LDS on semi-supervised classification problems for which a graph is not available.
* Analyzed the learned graph generative model to understand to what extent LDS is able to learn meaningful edge probability distributions even when a large fraction of edges is missing

1. Datasets

   * Cora & Citeseer: all edges are removed
   * Scikit-learn
   * FMA

2. Setup and Baselines

   * LDS | GCN | GCN-RND

   * Sparse-GCN | Dense-GCN | RBF-GCN | ...

     Two layers GCN + 16 hidden neurons + ReLu + Softmax

     Regularized cross-entropy loss: $L(w,A)=-\sum_{v\in V_{Train}} y_v \cdot \log[f_w(X,A)_w]+\rho ||w_1||^2$

     Dropout with $\beta = 0.5$

     Adam

     tunning the learning rate $\gamma$ from {$0.005,0.01,0.02$}

   * split the validation set: validation set | early stopping set

     Unregularized cross-entropy loss 

     Adam | SGD

     ...

## Related Work

1. Semi-supervised learning: only use labelled data to calculate loss

2. Graph synthesis and generation

3. Link prediction

4. Gradient estimation for discrete random variables

## Conclusion

* LDS, a framework that simultaneously learns the graph structure and the parameters of a GNN

* LDS's high accuracy on typical semi-supervised classiﬁcation datasets

* The edge parameters have a probabilistic interpretation

* Limitations:

  * cannot currently scale to large datasets

  * evaluate LDS only in the transductive setting

  * do not currently enforce the graphs to be connected when sampling

## Appendix

1. Extended algorithm

   * $w_{\theta,t+1}=\Phi(w_{\theta,T},A_t)$

   * $D_t:= \partial_w\Phi(w_{\theta,t},A_t)$

     $E_t:= \partial_A\Phi(w_{\theta,t},A_t)$

2. On the Straight-through Estimator

   Bernoulli: $\hat{g}(z)=\frac{\partial h(z)}{\partial z},z\text{~}P_\theta$

   If $h(z)=\frac{(az-b)^2}{2}, \frac{\partial}{\partial\theta}\mathbb{E}_{z\text{~}Ber(\theta)}[h(z)]=\frac{a^2}{2}-ab$ while $\mathbb{E}_{z\text{~}Ber(\theta)}[\hat{g}(z)]=\theta{a^2}-ab$, biased for $\theta \neq \frac{1}{2}$

