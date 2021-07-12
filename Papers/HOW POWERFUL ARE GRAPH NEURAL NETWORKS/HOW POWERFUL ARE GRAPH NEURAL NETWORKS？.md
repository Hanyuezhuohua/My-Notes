# HOW POWERFUL ARE GRAPH NEURAL NETWORKS？

## Abstruct

* GNN: a neighborhood aggregation scheme

  representation vector of a node: recursively aggregate and transform representation vectors of its neighboring nodes 

* a theoretical framework for analyzing the expressive power of GNNs to capture different graph structures

* develop a simple architecture that is provably the most expressive among the class of GNNs and is as powerful as the Weisfeiler-Lehman graph isomorphism test

## Introduction

* GNN: neighborhood aggregation + graph-level pooling schemes

* formally characterize how expressive different GNN variants are in learning to represent and distin- guish between different graph structures

* inspired by the close connection between GNNs and the Weisfeiler-Lehman (WL) graph isomorphism test

* WL: injective aggregation update that maps different node neighborhoods to different feature vectors

* Definition: an aggregation function over the multiset

* Main results:

  * show that GNNs are *at* *most* as powerful as the WL test in distinguishing graph structures
  * We establish conditions on the neighbor aggregation and graph readout functions under which the resulting GNN is *as* *powerful* *as* the WL test
  * identify graph structures that cannot be distinguished by popular GNN variants, such as GCN and GraphSAGE, and we precisely characterize the kinds of graph structures such GNN-based models can capture
  * We develop a simple neural architecture, *Graph* *Isomorphism* *Network* *(GIN)*, and show that its discriminative/representational power is equal to the power of the WL test

## Preliminaries

* Let $G = (V, E)$ denote a graph with node feature vectors $X_v$ for $v \in V$

* two tasks:

  * Node classification: each node $v \in V$ has an label $y_v$ + learn a representation vector $h_v$ of $v$ predicting $y_v = f(h_v)$
  * Graph classification: given a set of graphs $\{G_1,...,G_N\} \subseteq \mathcal{G}$ and their labels $\{y_1,...,y_n\} \subseteq \mathcal{Y}$ + learn a representation vector $h_G$ predicting $y_G = g(h_G)$

*  $a_v^{(k)}=\text{AGGREGATE}^{(k)}(\{h_u^{(k-1)}:\forall u\in\mathcal{N}(v)\}),\quad h_v^{(k)}=\text{COMBINE}^{(k)}(h_v^{(k-1)},a_v^{(k)})$

  initialize $h_v^{(0)}=X_v$

* In the pooling variant of GraphSAGE: 

  * $a_v^{k}=\text{MAX}(\{\text{ReLU}(W\cdot h_U^{(k-1)}),\forall u \in \mathcal{N}(v)\})$
  * $h_v^{(k)}=W \cdot [h_v^{(k-1)},a_v^{(k)}]$

  In Graph Convolution: $h_v^{(k)}=\text{ReLU}(W \cdot \text{MEAN}\{h_u^{(k-1)},\forall u\in\mathcal{N}(v)\cup\{v\}\})$  

* Graph classification: $h_G =\text{READOUT}(\{h_v^{(K)} |\ v\in G\})$

* Weisfeiler-Lehman test: $h_l^{(t)}(v)=\text{HASH}(h_l^{(t-1)}(v),F\{h_l^{(t-1)}|\forall u\in \mathcal{N}(v)\})$

  <https://blog.csdn.net/qq_34138003/article/details/108172823>

  WL subtree kernel: a node's label at $k$-th iteration of WL test represents a subtree structure of height $k$ rooted at the node.

## Theoretical framework: overview

