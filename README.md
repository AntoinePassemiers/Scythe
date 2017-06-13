# Scythe

[Under development] Deep learning library based on random forests

## Features

|         | C (fit) | C (predict) | R (fit) | R (predict) |
|---------|:-------:|:-----------:|:-------:|------------:|
| RF      | done    | done        |         |             |
| CRF     | done    | done        |         |             |
| GB      | ?       | ?           | ?       | ?           |

## Deep forests

The idea is partly inspired by the original deep forest article:


Deep Forest: Towards An Alternative to Deep Neural Networks

Zhi-Hua Zhou, Ji Feng

arXiv:1702.08835 [cs.LG]

https://arxiv.org/abs/1702.08835

### Todo

- [x] [CART/RF] Select a subset of features when selecting a split
- [ ] [CART] Set the minimum number of instances per leaf as a parameter
- [ ] Being able to pass non-contiguous Numpy arrays to the C API (not sure if necessary)
- [ ] Still use gradient boosting ?
- [ ] [RF] Adapt random forests' code for regression tasks
- [ ] Proper setup.py file
- [ ] Proper Makefile: fix fPIC error for Linux users

- [ ] Python wrapper of the LayerConfig structure

## License

Copyright © 2017 Scythe

Distributed under the Eclipse Public License either version 1.0.