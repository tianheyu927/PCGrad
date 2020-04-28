
# PCGrad

This repository contains code for [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) in TensorFlow v1.0+ (PyTorch implementation forthcoming).

PCGrad is a form of gradient surgery that projects a task’s gradient onto the normal plane of the gradient of any other task that has a conflicting gradient, which achieves substantial gains in efficiency and performance on a range of supervised multi-task learning and multi-task reinforcement learning domains. Moreover, it is model-agnostic and can be combined with previously-proposed multitask architectures for enhanced performance.

## Installation
Clone this repo and copy `PCGrad_tf.py` to your codebase.

## Usage

```python
optimizer = PCGrad(tf.train.AdamOptimizer()) # wrap your favorite optimizer
losses = # a list of per-task losses
assert len(losses) == num_tasks
train_op = optimizer.minimize(losses)
```

## Experiments

Our experiments in the paper were based on the following repositories.

CIFAR-100-MTL: [RoutingNetworks](https://github.com/cle-ros/RoutingNetworks)

NYUv2: [MTAN](https://github.com/lorenmt/mtan)

MultiMNIST: [MultiObjectiveOptimization](https://github.com/intel-isl/MultiObjectiveOptimization)

MT10/MT50/goal-conditioned pushing in [MetaWorld](https://meta-world.github.io/): [softlearning](https://github.com/rail-berkeley/softlearning) with modifications (per-task temperature and per-task replay buffers). We will release mofified multi-task softlearning code soon.


## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}
```
