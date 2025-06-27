# torch-efa
PyTorch implementation of Euclidean Fast Attention (EFA) and Euclidean RoPE (ERoPE)

## Euclidean Fast Attention

[EFA](https://arxiv.org/abs/2412.08541) is an O(3)-equivariant, linear self-attention mechanism for 3D geometric data like molecules. Using kernelized attention, EFA can achieve all-to-all message passing. The work also introduces Euclidean Rotary Positional Encodings (ERoPE), a mechanism to encode orientational information between pairs of nodes directly into the node features. 

$$
\operatorname{EFA}(\mathbf{h}_i, \vec{\mathbf{x}}_i) = \frac{1}{4\pi}\int_{S^2} \phi_{\vec{\mathbf{u}}}(\vec{\mathbf{x}}_i, \psi(\mathbf{q}_i))^\top\sum_{n=1}^{N} \phi_{\vec{\mathbf{u}}}(\vec{\mathbf{x}}_n, \psi(\mathbf{k}_n))\mathbf{v}^\top_n \otimes Y^{(l)}_m(\vec{\mathbf{u}})~~d\vec{\mathbf{u}}
$$

The surface integral on the sphere is approximated using [Lebedev Quadrature](https://en.wikipedia.org/wiki/Lebedev_quadrature).

> `torch-efa` is a fast and scalable PyTorch implementation of EFA and ERoPE for ML-based interatomic potentials.

## Installation

You can install `torch-efa` via `uv` and `pip`:

```bash
$ uv pip install torch-efa
```

You can also build the package from source:

```
bash
$ git clone https://github.com/rish-16/torch-efa.git
$ cd torch-efa
$ uv sync
```

## Usage

You can use the `EuclideanFastAttention` module like any other `torch.nn.Module` in place of existing equivariant