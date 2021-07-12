# Tutorial for Deep Learning

**Deep Learning:**   $\text{AI}\Rightarrow\text{ML}\Rightarrow\text{RL}$

## The foundation of neural networks

* $\text{Bernoulli} \Rightarrow \text{sigmoid}$

  推导： 

  * 指数分布族：$p(y;\eta) = b(y)e^{\eta^TT(y)-a(\eta)}$
  * 伯努利分布：$p(y;\phi)=\phi^{y}(1-\phi)^{1-y}=e^{ylog\phi+(1-y)\log(1-\phi)}=e^{\log(\frac{\phi}{1-\phi})y+\log(1-\phi)}$
  * 两者比较得到：
    * $\eta=\log(\frac{\phi}{1-\phi}) \Rightarrow \phi = \frac{1}{1+e^{-\eta}}$
    * $T(y) = y$
    * $a(\eta)=-\log(1-\phi)=\log(1+e^\eta)$
    * $b(y)=1$
  * 方法一：$h(x)=E(T(y))=E(y)=\phi=\frac{1}{1+e^{-\eta}}$
  * 方法二：指数分布族的数学期望是对数分布函数，$a(\eta)$一阶偏导 $h(x)=E(T(y))=E(y)=a'(\eta)=\frac{1}{1+e^{-\eta}}$
  * 广义线性模型性质：自然参数 $\eta$ 和 $x$ 满足线性关系 $\eta=\theta^{T}x$

* $KL(\mathbb{P_{d}, P_{m}}) = \mathbb{E}_{(x,y)\text{~}p(x,y)}[\log\frac{p(x,y)}{p_m(x,y)}]=\mathbb{E}_{x\text{~}p(x),y\text{~}p(y|x)}[\log p(x)p(y|x)-log(p(x)p_\theta(y|x))]$

  二分类：$$\underset {\theta}\min KL(\mathbb{P_{d}, P_{m}})=\underset {\theta}\min \frac{1}{N}\sum_{i=1}^{n}-y_i\cdot\log p_{\theta}(y=y_i|x_i)-(1-y_i)\cdot\log(1-p_{\theta}(y=y_i|x_i)$$

* $\text{Multinomial} \Rightarrow \text{softmax}$

  $p_\theta(y=k|x)=\frac{e^{f_\theta(x)_k}}{\sum_{m=1}^{K}e^{f_\theta(x)_m}}$

* 多分类：$$\underset {\theta}\min KL(\mathbb{P_{d}, P_{m}}) = \underset {\theta}\min L(\theta) = \underset {\theta}\min -\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(y=y_i|x_i)$$

* $\text{Regression}：\text{Gaussian distribution } \mathbb{P}_m = N(y|x;\mu,\sigma) \text{ where } \mu(x)=f_\theta(x)$
  $$
  \begin{align*}
  &\quad\underset {\theta}\min KL(\mathbb{P_{d}, P_{m}})\\
  &=\underset {\theta}\max \int_xp(x)\int_y\log p_\theta(y|x)dydx\\
  &=\underset {\theta}\max \frac{1}{N}\sum_{i=1}^{N}\int_yp(y|x_i)\log \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y-\mu(x))^2}{2\sigma^2}}dy\\
  &= \underset {\theta}\min\frac{1}{N}\sum_{i=1}^{N}(y_i-\mu(x))^2
  \end{align*}
  $$

* $\text{Back-Propagation Gradient Computation}$
  $$
  \begin{align*}
  \text{Lemma 1: }
  F_s(\theta)&=||Y-UVX||_2^2 \Rightarrow \frac{\partial F_s(\theta)}{\partial V} = -2U^T(Y-UVX)X^T \\
  
  \text{Lemma 2: }
  F_s(\theta)&=||Y-U\phi(VX)||_2^2 \Rightarrow \frac{\partial F_s(\theta)}{\partial V} = -2DU^T(Y-U\phi(VX))X^T \quad\text{ where } D=diag\{\phi'(VX)\} 
  \end{align*}
  $$

  $$
  \begin{align*}
  \frac{\partial F_s(\theta)}{\partial W^l}&=-2(W^{L}\cdot D^{L-1}\cdot W^{L-1}\cdot D^{L-2}...W^{l+1}\cdot D^l)^{T}\cdot e \cdot z_{l-1}^T\\
  &=-2(W^{L}\cdot D^{L-1}\cdot W^{L-1}\cdot D^{L-2}...W^{l+1}\cdot D^l)^{T}\cdot e \cdot (\phi(W^{l-1}...W^2\phi(W^1x)...))^T\\
  \text{where } &D^l=diag\{\frac{\partial\phi(h_i^l)}{\partial h_i^l}\}_{i=1}^{d_l} \\
  \text{Let }g(l) &= W^{L}\cdot D^{L-1}\cdot W^{L-1}\cdot D^{L-2}...W^{l+1}\cdot D^l \Rightarrow g(l)=g(l+1)\cdot W^{l+1}\cdot D^l\\
  f(l) &= \phi(W^{l-1}...W^2\phi(W^1x)...) \Rightarrow f(l)=\phi(W^lf(l-1))\\
  \text{Then D} &\text{ynamic Programming}
  \end{align*}
  $$

* **Protocol for ML/DL Tasks: ** Training | Validation | Test

## Tutorial on Pytorch using Colab platform

**Pytorch：** <https://pytorch.org/> | <https://pytorch.org/docs/>

### Learn the basics

<https://pytorch.org/tutorials/beginner/basics/intro.html>

1. QUICKSTART

   * Working with data 

     * `torch.utils.data.Dataset`

       Example: `training_data = datasets.FashionMNIST()`

       Tips: `torchvision.transforms.ToTensor` + `transform = ToTensor()`

       * $\text{PIL Image | ndarray}\text{ (H, W, C)} \Rightarrow \text{tensor} \text{ (C, H, W)}$

       * $ \textbf{[}0\text{ - }1\textbf{]}\text{ normalization} $

     * `torch.utils.data.DataLoader`

       Example: `train_dataloader = DataLoader(training_data, batch_size=batch_size)`
       
     * Other: <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>
   
   * Creating Models
   
     * Get cpu or gpu device
   
       `device = "cuda" if torch.cuda.is_available() else "cpu"`
   
     * `torch.nn.Module`
   
       Example:
   
       ```python
       # Define model
       class NeuralNetwork(nn.Module):
           def __init__(self):
               super(NeuralNetwork, self).__init__()
               self.flatten = nn.Flatten()
               self.linear_relu_stack = nn.Sequential(
                   nn.Linear(28*28, 512),
                   nn.ReLU(),
                   nn.Linear(512, 512),
                   nn.ReLU(),
                   nn.Linear(512, 10),
                   nn.ReLU()
               )
       
           def forward(self, x):
               x = self.flatten(x)
               logits = self.linear_relu_stack(x)
               return logits
       
       model = NeuralNetwork().to(device)
       ```
   
     * Print Model Parameters: 
   
       ```python
       for name, param in model.named_parameters():
           print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
       ```
   
     * Other: <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html> 
   
   * Optimizing the Model Parameters
   
     * Loss Function: `loss_fn = nn.CrossEntropyLoss()`
   
     * Optimizer: `optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)`
   
     * Backpropagation:
   
       ```python
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       ```
   
     * Other: `https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html`
   
   * Saving Models: `torch.save(model.state_dict(), "model.pth")`
   
   * Loading Models:
   
     ```python
     model = NeuralNetwork()
     model.load_state_dict(torch.load("model.pth"))
     ```
   
     * Test: `model.eval()` + `with torch.no_grad():`
   
2. Tensors

   * `torch.tensor(data)` | `torch.Tensor(data)`
   * `torch.from_numpy(np_array)`
   * `torch.ones_like(x_data)` | `torch.rand_like(x_data, dtype=torch.float)`
   * `torch.rand(shape)` | `torch.ones(shape)` | `torch.zeros(shape)`

3. Transforms

   * `torchvision.transforms.Lambda` + `torch.zeros.scatter_()`

     ```python
     target_transform = Lambda(lambda y: torch.zeros(
         10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))     
     ```
     **one-hot**

4. Autograd

   ```python
   import torch
   
   x = torch.ones(5)  # input tensor
   y = torch.zeros(3)  # expected output
   w = torch.randn(5, 3, requires_grad=True)
   b = torch.randn(3, requires_grad=True)
   z = torch.matmul(x, w)+b
   loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
   loss.backward() //if loss is a scalar，
   //like loss.backward(torch.FloatTensor([1.0, 1.0, 1.0]) if not scalar
   ```

   `detach()// requires_grad=False & grad_fn=False` 

5. Save & Load Model

   * Saving and Loading Models with Shapes （no weight）

     ```python
     torch.save(model, 'model.pth')
     model = torch.load('model.pth')
     ```

   * Exporting Model to ONNX
   
     ```python
     input_image = torch.zeros((1,3,224,224))
     onnx.export(model, input_image, 'model.onnx')
     ```

### LEARNING PYTORCH WITH EXAMPLES

<https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples-download>

* `torch.float`
* `torch.device("cpu")` | `torch.device("cuda:0")` | `.to(device)` | `.cpu()`
* `torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)`
* `torch.sin(x)`
* `torch.randn((), device=device, dtype=dtype)`
* `tensor.item()`
* `class LegendrePolynomial3(torch.autograd.Function)` new autograd function
* `torch.optim.lr_scheduler.StepLR`

## Rethinking Neural Network Common Issues

* Mini-batch SGD Training

* Parallelization

* Initialization: Xavier Glorot Initialization | Kaiming Initialization: 

  $\mathbb{E}[W]=0, \mathbb{V}[W]=\frac{1}{\sqrt{d}}$

* Batch Normalization: $W^l \Rightarrow BN \Rightarrow\phi$

* ResNet Architecture:$output = \mathcal{F}(input)+ input$

* Universal Approximation Theorem

* Error: Representation error | Optimization error | Generalization error

  Training error = representation error + optimization error

  Test error = training error + generalization error

* Generalization Error Analysis

  * expectation risk: $R(f)=\mathbb{E}_{(x,y)\text{~}p(x,y)}[L(y, f(x))]$

    Empirical risk: $\hat{R}(f)=\frac{1}{N}\sum_{i=1}^NL(y_i), f(x_i)$

  * Generalization Error Analysis: $R(f) \leq \hat{R}(f) + \epsilon(d,N,\delta)$ where $\epsilon(d, N, \delta = \sqrt{\frac{1}{2N}(\log d + \log\frac{1}{\delta})}$

    Proof: <https://blog.csdn.net/qq_43872529/article/details/104362791>

* Over-Fitting and Regularization

  * Weight dacaying regularization

    * MLE: $\underset{\theta}{\arg \max} \text{ } p(D|\theta)$

    * MAP: $\underset{\theta}{\arg \max} \text{ } p(D|\theta)p(\theta)$

      <https://blog.csdn.net/andyelvis/article/details/42423185>
      $$
      \begin{align*}
      &\text{Assume model parameter follows a zero-mean Gaussian distribution}\\
      &\underset{\theta} \max \sum_{i=1}^N \log p(x_i)p_{\theta}(y_i|x_i) + \log p_0(\theta)\\
      = &\underset{\theta} \max \sum_{i=1}^N \log \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}} + \log \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{\theta^2}{2\sigma_0^2}}\\
      = &\underset{\theta} \min -\sum_{i=1}^N|y_i-f(x_i)|^2+\frac{\sigma^2}{\sigma_0^2}||w||_2^2
      \end{align*}
      $$
      
  
  * Random dropout neurons
  
  * Use small neural network（not use frequently）

## Variants of Neural Networks

* Convolutional Neural Network

  * Input: a batch of images [N, W, H, C] = [nums, width, height, channel]

    Output: feature maps [N, W', H', C']

  * Parameter: kernel matrix weight [w, h, c_in, c_out], bias vector [c_out]

  * **1 CNN layer = 1 convolution layer ( + 1 pooling layer)**

  * $\text{image feature} \bigotimes \text{kernel matrix} = \text{output feature map}$

  * padding | stride

  * Implementation:

    * `torch.nn.Conv2d()`
    * `torch.nn.functional.conv2d()`
    * `torch.nn.MaxPool2d()`
    * `torch.nn.functional.max_pool2d()`

  * Transposed Convolution

    $Cx=y \Rightarrow C^Ty=x'$

* Recurrent Neural Network

  * Input: a batch of sequences [N, T, H] = [num, length, features], initial hidden state [N, H]

  * Output: hidden states [N, T, H'], output state [N, H']

  * Parameter: $\text{W}_\text{h}$ [H, H'], $\text{W}_\text{x}$ [H, H'], $\text{b}_\text{h}$ [H'], $\text{b}_\text{x}$ [H']

  * $h_t = tanh(W_hh_{t-1}+W_xx_t+b_h+b_x), \hat{y_t}= softmax(W_yh_t+b_y)$

  * Back-Propagation Through Time：

    * $\frac{\partial\hat{y}_T}{\partial W_h} = \frac{\partial\hat{y}_T}{\partial h_T} \cdot \sum_{t=1}^{T}(\prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} \cdot \frac{\partial h_t}{\partial W_h})$
    * $\frac{\partial\hat{y}_T}{\partial W_x} = \frac{\partial\hat{y}_T}{\partial h_T} \cdot \sum_{t=1}^{T}(\prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} \cdot \frac{\partial h_t}{\partial W_x})$

  * Bidirection RNN

  * Limitation of RNN

    * gradient Vanishing/explosion:

      $\prod_{k=1}^T\frac{\partial h_k}{\partial h_{k-1}} = W_h^T =(U\sum V)^T=U\sum^TV=\sum_{i=1}^r\sigma_i^Tu_iv_i^{Transpose}$

      if too long, consider $\sigma_i$

   * Solution: LSTM | GRU

     * GRU:
       $$
       \begin{align*}
       z_t&=\sigma(W_z \cdot [h_{t-1},x_t]\\
       r_t&=\sigma(W_r \cdot [h_{t-1},x_t])\\
       \tilde{h_t}&=\tanh(W \cdot [r_t * h_{t-1}, x_t])\\
       h_t &= (1-z_t)*h_{t-1} + z_t * \tilde{h_t}
       \end{align*}
       $$
       

   * Implementation:

     * `torch.nn.RNN`
     * `torch.nn.RNNCell`

* Graph Neural Network & Graph Convolution Network

  * Laplacian operator

    $\Delta f = \nabla ^2f = \sum_{i=1}^n\frac{\partial^2f}{\partial x_i^2}$

    $\frac{\partial^2f}{\partial x_i^2} \approx f'(x)-f'(x-1) \approx f(x+1) + f(x-1) - 2f(x)$

  * Laplacian operator on graph

    $$
    \begin{align*}
    &\Delta f(i) = \sum_{j\in N_i}\frac{\partial ^2 f(i)}{\partial j^2} \approx \sum_j a_{ij}(f(i)-f(j)) = d_if(i)-A_i\textbf{f}\\
    &\Delta \textbf{f} = 
    \left [
    \begin{matrix}
    \Delta f(1)\\
    \Delta f(2)\\
    ...\\
    \Delta f(n)
    \end{matrix}
    \right]
    =
    \left [
    \begin{matrix}
    d_1f(1)-A_1\textbf{f}\\
    d_2f(2)-A_2\textbf{f}\\
    ...\\
    d_nf(n)-A_n\textbf{f}
    \end{matrix}
    \right]
    =(D-A)\cdot\textbf{f}
    =L \cdot \textbf{f}\\
    &D(i,j)=\left\{\begin{matrix}d_i，&\text{if } i =j\\0，&\text{otherwise} \end{matrix}\right.\\
    &A(i,j)=\left\{\begin{matrix}1，&\text{if } i \in N_j \text{ and } j \in N_i\\0，&\text{otherwise} \end{matrix}\right.
    \end{align*}
    $$
  
  * Fouier transform on graph $\Rightarrow$ Convolution in graph
  
    <https://blog.csdn.net/qq_41727666/article/details/84622965>
  
    $(f * g)_\mathcal{G} = UH(\Lambda)U^Tf$
  
  * Conclusion:
  
    * $H^{(1)} = \sigma(W^1(I+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})X)$
    * $H^{(l+1)} = \sigma(W^1(I+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})H(l))$
    * $Y=f(H^{(l+1)})$
  
  * Graph attention networks | GraphSAGE | Message Passing
  
  * Implementation
  
    * `import torch_geometric.nn`
    * `GCNConv()`
  
* Transformer: Attention | Positional Embedding
  

