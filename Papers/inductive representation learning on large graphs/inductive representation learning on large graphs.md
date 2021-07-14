# inductive representation learning on large graphs

## Abstract

* Existing approach: require all nodes presenting during training $\rightarrow$ transductive

  do not generalize to unseen nodes

* GraphSAGE: a general inductive framework that leverage node feature information to efficiently generate node embeddings for previously unseen data

* learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood

## Introduction

* Node embedding approaches:  use dimensionality reduction techniques to distill the high-dimensional information about a node’s graph neighborhood into a dense vector embedding

* Transductive vs Inductive: evolving graphs | unseen nodes | generalizationa across graph

* Generalizing to unseen nodes requires "aligning"

* An inductive framework must learn to recognize structural properities of a node's neighborhood: the node's local role in the graph & its global position

* Extend GCNs to the task of inductive unsupervised learning 

  Propose a framework that generalized the GCN approach to use trainable aggregation functions

* learn the topological structure of each node's neighborhood as well as the distribution of node features in the neighborhood

* train a set of aggregator function instead of a distinct embedding vector for each node

## Related work

1. Factorization-based embedding approaches

   * use random walk statistics and matrix factorization-based learning objectives
   * directly train node embeddings for individual nodes $\rightarrow$ transductive $\rightarrow$ require expensive additional training for new nodes
   * However, our model leverage feature information in order to train a model produce embeddings for unseen nodes

2. Supervised learning over graphs

   * a wide variety of kernel-based approaches
   * Previous work: classify entire graph ｜ This work: generate useful representations for individual nodes

3. Graph convolutional networks

   * A simple variant of our algorithm can be viewed as an extension of the GCN framework to the inductive setting

## Proposed method: GraphSAGE

**Key idea: learn how to aggregate feature information from a node's local neighborhood**

1. Embedding generation algorithm

   ![截屏2021-07-14 16.03.48](/Users/malachite/Library/Application Support/typora-user-images/截屏2021-07-14 16.03.48.png)
   
   Extend Algorithm 1 to the minibatch setting:
   
   * first forward sample the required neighborhood sets(up to depth $K$)
   
   * then run the inner loop
   
   ![截屏2021-07-14 16.15.10](/Users/malachite/Library/Application Support/typora-user-images/截屏2021-07-14 16.15.10.png)
   
2. 

