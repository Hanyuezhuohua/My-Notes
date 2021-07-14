# Semi-supervised classification with graph convolutional networks

## Abstract

* convolutional architecture via a localized first-order approximation of spectral graph convolutions
* scale linearly in the number of graph edges
* learn hidden layer representations that encode both local graph structure and features of nodes

## Introduction

* semi-supervised learning: labels are only available for a small subset of nodes

  label information is smoothed over the graph via some form of explicit graph-based regularization: (by using a graph Laplacian regularization term in the loss function)
  $$
  \mathcal{L} = \mathcal{L}_0 + \lambda\mathcal{L}_{reg}, \text{ with } \mathcal{L}_{reg} = \sum_{i,j}A_{i,j}||f(X_i)-f(X_j)||^2=f(X)^{\top}\Delta f(X)
  $$
  Here, $\mathcal{L}_0$ denote the supervised loss of the labeled part of the graph, $f(\cdot)$ can be a neural network-like differentiable function, $\Delta$ denots the unnormalized graph Laplacian.

  Problem: graph edges need not necessarily encode node similarity.

* encode the graph structure directly using a neural network model $f(X,A)$

  train on a supervised target $\mathcal{L}_0$ for all nodes with labels

  condition $f(\cdot)$ on the adjacency matrix of the graph will allow the model to distribute gradient information from the supervised loss $\mathcal{L}_0$ and enable it to learn representations of nodes both with and without labels

* Contribution:

  * introduce a simple and well-behaved layer-wise propagation rule for neural network models

  * demonstrate how this form of a graph-based neural network model can be used for fast and scalable semi-supervised classification of nodes in a graph 

## Fast approximate convolutions on graphs

consider a multi-layer Graph Convolutional Network with teh following layer-wise propagation rule:
$$
H^{(l+1)}=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{l})
$$
Here, $\tilde{A} = A + I_N, \tilde{D}_{ii}=\sum_j\tilde{A}_{ij}, W^{(l)}$ is a layer-specific trainable weight matrix$, \sigma(\cdot)$ denotes an activation function and $H^{(l)} \in \R^{N \times D}$ is the matrix of activations in the $l^{th}$ layer ($H^{(0)}=X$) 

1. Spectral graph convolutions

   * some basic information for spectral graph convolutions:

     from <https://blog.csdn.net/qq_41727666/article/details/84622965>
     $$
     (f*h)_G=U\left(
      \begin{matrix}
     
        \hat h(\lambda_1) &  &   \\
         & \ddots &  \\
         &  & \hat h(\lambda_n)
       \end{matrix} 
     \right)U^{\top}f
     $$
     第一代GCN：

     * 卷积核：$diag(\hat h(\lambda_l)): diag(\theta_l)$

     * output公式：
       $$
       y_{output} = \sigma\left(U\left(
        \begin{matrix}
       
          \theta_1 &  &   \\
           & \ddots &  \\
           &  & \theta_n
         \end{matrix} 
       \right)U^{\top}x\right)
       $$

     第二代GCN：

     * 卷积核：$\hat h(\lambda_l): \sum_{j=0}^{K}a_j\lambda_l^i$

     * output公式：
       $$
       y_{output} = \sigma\left(U\left(
        \begin{matrix}
       
          \sum_{j=0}^{K}a_j\lambda_1^j&  &   \\
           & \ddots &  \\
           &  & \sum_{j=0}^{K}a_j\lambda_n^j
         \end{matrix} 
       \right)U^{\top}x\right)
       =\sigma(\sum_{j=0}^Ka_jL^jx)
       $$

   * previous spectral graph convolutions in this paper: $g_\theta * x = U g_\theta U^\top x$ (第一代GCN)

     * $x \in \R^N$ is a scalar for every node
     * $g_\theta = diag(\theta)$ parameterized by $\theta \in \R^N$ in the Fourier domain
     * $L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=U\Lambda U^{\top}$

   * use a truncated expansion in terms of Chebyshev polynomials $T_k(x)$ up to $K^{th}$ order to well-appoximate $g_\theta(\Lambda)$: $g_{\theta'}(\Lambda)\approx \sum_{k=0}^{K}\theta'_kT_k(\tilde\Lambda))$

     with a rescaled $\tilde \Lambda = \frac{2}{\lambda_{max}}\Lambda-I_N$

     Then, $g_{\theta'} * x \approx \sum_{k=0}^K\theta'_kT_k(\tilde L)x$  with $\tilde L = \frac{2}{\lambda_{max}}L-I_N$

2. Layer-wise linear model

   * stack multiple above-mentioned layers($K=1$) + each layer followed by a point-wise non-linearity.

   * approximated $\lambda_{max} \approx 2$ and get $g_{\theta'} * x \approx \theta'_0x + \theta'_1 (L-I_N)x = \theta'_0x - \theta'_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x$ 

   * constrain the number of parameter further: $\theta = \theta'_0 = -\theta'_1$

     then $g_{\theta'} * x \approx \theta(I_N +D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$ 

   * in order to alleviate exploding/vanishing gradients:
     $$
     I_N +D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \rightarrow \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}\\
     \tilde A = A + I_N\quad
     \tilde D_{ii}=\sum_{j}\tilde A_{ij}
     $$

   * Generalize this definition to $X \in \R^{N \times C}$ (a $C$-dimensional feature vector to every node) and $F$  filters or feature maps as follows:
     $$
     Z = \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}X\Theta\\
     \Theta \in \R^{C\times F} \text{ is a matrix of filer parameters}\\
     \text{Complexity:} \mathcal{O}(|E|FC) 
     $$
     

## Semi-supervised node classification

expect the setting of $f(X,A)$ to be especially powerful in scenarios where the adjacency matrix contains information not present in the data $X$.

1. Example

   * $\hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde A \tilde{D}^{-\frac{1}{2}}$
   * $Z = f(X,A) = \text{softmax}(\hat{A}\text{ReLU}(\hat{A}XW^{(0)})W^{(1)})$
   * $\mathcal{L} = - \sum_{l \in \mathcal{Y}_L}\sum_{f=1}^{F}Y_{lf}\ln Z_{lf}$, where $\mathcal{Y}_L$ is the set of node indices that have labels
   * gradient descent + full dataset as batch + a sparse representation for $A$ + dropout

2. Implementation

   The computational complexity: $\mathcal{O}(|E|CHF)$

## Discussion

1. Limitations and future work

   * Memory requirement

   * Directed edges and edge feature

   * Limiting assumption

## Appendix

1. relation to weisfiler & Lehmann algorithm

   $h_i^{(t+1)} = \text{hash}(\sum_{j \in \mathcal{N}_i}h_J^{(t)}) \rightarrow h_i^{(t+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)}\right)$

   By taking $h_i^{(l)}$ to be a vector of activations of node $i$ in the $l^{th}$ neural network layer and choosing $c_{ij} = \sqrt{d_id_j}$, equals to $H^{(l+1)}=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$

   * node embeddings with random weight

     even an untrained GCN model with random weights can serve as a powerful feature extractor for nodes in a graph:
     $$
     Z = \text{tanh}(\hat{A}\text{tanh}(\hat{A}\text{tanh}(\hat{A}XW^{(0)})W^{(1)})W^{(2)})
     $$
     

   * semi-supervised node embeddings

     * only a single  labeled example pre class
     * $\text{softmax}(\text{tanh}(\hat{A}\text{tanh}(\hat{A}\text{tanh}(\hat{A}XW^{(0)})W^{(1)})W^{(2)}))$

2. experiment on model depth

   $H^{(l+1)}=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{l}) + H^{(l)}$

