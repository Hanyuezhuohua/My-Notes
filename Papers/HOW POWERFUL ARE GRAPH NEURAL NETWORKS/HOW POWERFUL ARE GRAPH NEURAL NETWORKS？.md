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

* assume node input features are from a countable universe

  node features from deeper layers of any fixed model are also from a countable universe

  assign each feature vector a unique label in $\{a,b,c...\}$

* Definition(Multiset): a multiset is a 2-tuple $X=(S,m)$ where $S$ is the underlying set of $X$ that is formed from its $distinct\ elements$, and $m: S \rightarrow \N_{\geq 1}$ gives the $multiplicity$ of the elements

* never map two neighborhoods to the same representation $\Rightarrow$ *injective* aggregation

## Building powerful graph neural networks

* Lemma 2: Let $G_1$ and $G_2$ be any two non-isomorphic graphs. If a graph neural network $\mathcal{A}:\mathcal{G}\rightarrow\R^d$  maps $G_1$ and $G_2$ to different embeddings, the Weifeiler-Lehman graph isomorphism test also decides $G_1$ and $G_2$ are not isomorphic.

* Theorem 3: Let $\mathcal{A}:\mathcal{G}\rightarrow\R^d$ be a GNN. With a sufficient number of GNN layers, $\mathcal{A}$ maps any graphs $G_1$ and $G_2$ that the Weisfeiler-Lehman test of isomorphism decides as non-isomorphic, to different embeddings if the following conditions hold:

  * $\mathcal{A}$ aggregates and updates node features iteratively with
    $$
    h_v^{(k)}=\phi(h_v^{(k-1)},f(\{h_u^{(k-1)}:u\in \mathcal{N}(v)\}))
    $$
    where the function $f$, which operates on multisets, and $\phi$ are injective.

  * $\mathcal{A}$'s graph-level readout, which operates on the multiset of node features $\{h_v^{(k)}\}$, is injective.

* Lemma 4: Assume the input feature space, $\mathcal{X}$ is countable. Let $g^{(k)}$ be the function parameterized by a GNN's k-th layer for $k=1,...,L$, where $g^{(1)}$ is defined on multisets $X \subset \mathcal{X}$ of bounded size. The range of $g^{(k)}$, i.e., the space of node hidden features $h_v^{(k)}$, is also countable for all $k=1,...,L.$

* Distinguish different graphs $\Rightarrow$ capture similarity of graph structure

1. Graph isomorphism network(GIN)

   * Lemma 5: Assume $\mathcal{X}$ is countable. There exists a function $f:\mathcal{X}\rightarrow\R^n$ so that $h(X) = \sum_{x \in X}f(x)$ is unique for each multiset $X \sub \mathcal{X}$ of bounded size. Moreover, any multiset function $g$ can be decomposed as $G(X)=\phi (\sum_{x\in X}f(x))$ for some function $\phi$.

   * Corollary 6: Assume $\mathcal{X}$ is countable. There exists a function $f: \mathcal{X}\rightarrow \R^n$ so that for infinitely mant choices of $\varepsilon$, include all irrational numbers, $h(c, X) = (1+\varepsilon)\cdot f(c) + \sum_{x \in X}f(x)$ is unique for each pair $(c,X)$, where $c \in \mathcal{X}$ and $X \subset \mathcal{X}$ is a maltiset of bounded size. Moreover, ant function $g$ over such pairs can be decomposed as $g(c,X) = \varphi((1+\varepsilon)\cdot f(c)+\sum_{x \in X}f(x))$ for some function $\varphi$.

   * According to the universal approximation theorem:
     $$
     h_v^{(k)}=\text{MLP}^{(k)}((1+\varepsilon^{(k)})\cdot h_v^{k-1}+\sum_{u\in\mathcal{N}(v)}h_u^{(k-1)}).
     $$

2. Graph-level readout of GIN
   $$
   h_G = \text{CONCAT}(\text{READOUT}(\{h_v^{(k)}|v\in G\})\ |\ k=0,1,...,K).
   $$
   GIN replaces READOUT with summing all node features from the same iterations

## Less powerful but still intersting GNNs

* Two aspects of the aggregator in the equation
  * 1-layer perceptrons instead of MLPs
  * mean or max-pooling instead of the sum

1. 1-layer perceptrons are not sufficient

   * 1-perceptron $\sigma \circ W$: a linear mapping followed by a non-linear activation function such as a ReLU.
   * Lemma 7: There exist finite multisets $X_1 \neq X_2$ so that for any linear mapping $W$, $\sum_{x\in X_1}\text{ReLU}(Wx)=\sum_{x\in X_2}\text{ReLU}(Wx).$

2. Structures that confuse mean and max-pooling

3. Mean learns distributions

   Corollary 8: Assume $\mathcal{X}$ is countable. There exists a function $f:\mathcal{X}\rightarrow\R^n$ so that $h(X) = \frac{1}{|X|}\sum_{x\in X}f(x), h(X_1) = h(X_2)$ if and onlye if multisets $X_1$ and $X_2$ have the same distribution. That is, assuming $|X_2| \geq |X_1|$, we have $X_1 = (S,m)$ and $X_2 = (S, k \cdot m)$ for some $k \in \N_{\geq 1}.$

4. Max-pooling learns sets witj distinct elements

   Corollary 9: Assume $\mathcal{X}$ is countable. Then there exists a function $f:\mathcal{X}\rightarrow\R^{\infty}$ so that for $h(X) = \max_{x\in X}f(x), h(X_1)=h(X_2)$ if and only if $X_1$ and $X_2$ have the same underlying set.

5. Remarks on other aggregators

   * weighted average via attention
   * LSTM pooling

## Other related work

## Experiments

## Conclusion

