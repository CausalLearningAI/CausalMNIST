# CausalMNIST
New Benchmark for *downstream Treatment Effect Estimation from ML pipelines* 

## Problem
Starting from [MNIST dataset](http://yann.lecun.com/exdb/mnist/), we manipulated the background color $B$ of each image (1: green, 0: red), and the pen color $P$ (1: white, 0: black) to enforce the following Conditional Average Treatment Effect:
$$\mathbb{E}[Y|do(B=1), P=1]-\mathbb{E}[Y|do(B=0), P=1]=0.4$$
$$\mathbb{E}[Y|do(B=1), P=0]-\mathbb{E}[Y|do(B=0), P=0]=0.2$$
and Average Treatment Effect:
$$\mathbb{E}[Y|do(B=1)]-\mathbb{E}[Y|do(B=0)]=0.3$$
where $Y$ is a binary variable equal to 1 if the represented digit is strictly greater than $d \in \mathbb{R}$, 0 otherwise. We assumed Ignorability and considered only $n_s$ images annotated (with Y) out of the full 60 000 images. We aim to recover the missing annotation to estimate leverage the ignorability assumption and estimate the ATE.

## Structural Causal Model
#### Noises
$$n_B \sim Be(0.5)$$

$$n_P \sim Be(0.5)$$

$$n_X \sim P^{n_X}$$

$$n_Y \sim P^{n_Y}$$

#### Structural equations

$$B := n_B$$

$$P := n_P$$

$$X := f_1(B, P, n_X)$$

$$Y := f_2(X, n_Y)$$

where $P^{n_X}, P^{n_Y}, f_1$ and $f_2$ are unknown and characteristic of MNIST dataset. 


#### Example
![Example Image](./results/CausalMNIST/biased/example.png)