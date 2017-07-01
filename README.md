# Scythe

** Under development ** Deep learning library based on deep forests

A deep forest consists of a stack of layers, where each of them learns from both the input features and the output features of the previous layer. Each layer is made of a series of complete-random forests. This results in a whole hierarchy of predictors, from single trees to deep forests.

The rationale behind this project is to solve the problem of high dimensionality of fine grained scanners. Indeed, contrary to neural networks' convolutional layers, multi-grained scanners tend to increase the size of the temporary data due to slicing. To get around this issue, the concept of virtual datasets has been developed. Furthermore, fast tree learning algorithms have been implemented.

## Deep forests

The idea is partly inspired by the [original deep forest article](https://arxiv.org/abs/1702.08835):

* Deep Forest: Towards An Alternative to Deep Neural Networks, by Zhi-Hua Zhou and Ji Feng (arXiv:1702.08835 [cs.LG]) *

![alt text](https://raw.githubusercontent.com/AntoinePassemiers/Scythe/master/doc/imgs/gcForest.png)

### Todo

- [x] * CART/RF * Select a subset of features when selecting a split
- [ ] * CART * Set the minimum number of instances per leaf as a parameter
- [ ] Use gradient boosting ?
- [ ] * RF * Adapt random forests' code for regression tasks
- [ ] Proper setup.py file
- [ ] Proper Makefile: fix fPIC error for Linux users
- [x] Python wrapper of the LayerConfig structure