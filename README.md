# A Statistical Analysis of Polyak-Ruppert Averaged Q-learning

This repository contains the codes for the paper

> [Communication-Efficient Distributed SVD via Local Power Iterations](https://arxiv.org/pdf/2112.14582.pdf)

If you find this code useful in your research, please consider citing:

    @inproceedings{li2023statistical,
    title={A statistical analysis of {P}olyak-{R}uppert averaged {Q}-learning},
    author={Li, Xiang and Yang, Wenhao and Liang, Jiadong and Zhang, Zhihua and Jordan, Michael I},
    booktitle={International Conference on Artificial Intelligence and Statistics},
    pages={2207--2261},
    year={2023},
    organization={PMLR}
    }

## Table of Contents
|File name   | Description |
| :-----       | :----       |
|env.py        | Two used environments, namely **RandomMDP** and **SimpleMDP**|
|main.py         | Execution code in RandomMDP   |
|main_simple.py   | Execution code in SimpleMDP |
|algo.py     | Functions for considered algorithms and the statistical inference procedure.|
|vi.py   | Apply value iteration to estimate the optimal value function |
|plot_online.py | Code for producing Figure 1|
|plot_convergence.py     | Code for producing Figure 2 |


## Some notes
1\. Acceleration by Ray: 

Due to the need of repeated simulations, we use the distributed execution engine **Ray** to accelerate the fitting and save the experiment time. 
As a result, you may find the following code at the beginning of these main files:

```
import ray
ray.init(num_cpus=NUM_CPU, num_gpus=NUM_GPU)
res = ray.get(FUNCTION_TO_BE_PARALLELED)
```

2\. Detail about the SimpleMDP:

The SimpleMDP is borrowed from [[Wainwright 2019](https://arxiv.org/pdf/1905.06265)]. See Section 3.4 and Figure 1 therein. 


3\. Online statistical procedure for Q-Learning:

Once we establish the functional central limit theorem for the partial-sum process (Theorem 3.1 in our paper), we can apply Algorithm 1 in [[Lee et al., 2021](https://arxiv.org/abs/2106.03156)] to compute the pivotal statistic.

4\. Considered algorithms:

We consider two algorithms. One is the averaged Q learning (named as "aql" in the code), while the other is the entropy regularized variant (named as "entropy" in the code).

