# MLX

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**Examples**](#examples)

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

MLX is an array framework for machine learning on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy. MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API. MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
  transformations for automatic differentiation, automatic vectorization,
  and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
  materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
  dynamically. Changing the shapes of function arguments does not trigger
  slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
  (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
  is the *unified memory model*. Arrays in MLX live in shared memory.
  Operations on MLX arrays can be performed on any of the supported
  device types without transferring data.

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas.

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

## Benchmarks (this fork: `perf/cpu`)

> This fork's `perf/cpu` branch speeds up the **CPU backend** (Linux, x86).
> Measured on an 8 vCPU Intel Xeon VM (320 MiB L3, no GPU), GCC 13, upstream
> `96fbf67` vs `perf/cpu`. These are median times per call over 15 rounds of
> 10 calls. Metal / Apple Silicon were not measured. Full method, all ops,
> single-thread numbers and limits are in [PERF.md](PERF.md) and
> [docs/mlx-cpu-perf.pdf](docs/mlx-cpu-perf.pdf).

| Op (float32) | Upstream (ms) | perf/cpu (ms) | Speedup |
|---|---:|---:|---:|
| add f32 4096x4096 | 25.48 | 7.33 | 3.5x |
| add bcast row 4096x4096 | 22.57 | 6.65 | 3.4x |
| exp f32 4096x4096 | 258.69 | 9.81 | 26.4x |
| tanh f32 4096x4096 | 96.78 | 16.07 | 6.0x |
| where 4096x4096 | 111.50 | 15.86 | 7.0x |
| sum all 4096x4096 | 8.74 | 0.44 | 19.9x |
| sum axis=0 4096x4096 | 2.80 | 1.19 | 2.3x |
| max axis=-1 4096x4096 | 17.13 | 0.82 | 20.8x |
| softmax f32 4096x4096 | 277.83 | 12.94 | 21.5x |
| softmax f32 8x32x512x512 | 1093.90 | 50.26 | 21.8x |
| rms_norm 4096x4096 | 59.79 | 16.57 | 3.6x |
| layer_norm 4096x4096 | 109.27 | 25.79 | 4.2x |
| transpose copy 4096x4096 | 184.50 | 9.41 | 19.6x |
| swapaxes copy 8x32x512x512 | 170.87 | 32.97 | 5.2x |
| sort axis=-1 1024x4096 | 204.78 | 27.63 | 7.4x |
| argsort axis=-1 1024x4096 | 245.35 | 33.24 | 7.4x |
| gather rows 4096 of 32000x1024 | 1.28 | 1.24 | 1.0x |

![CPU speedups](docs/perf/cpu_speedup.png)

`gather` was not changed and is shown as a control. Reproduce with
`python benchmarks/python/cpu_ops_bench.py` (see PERF.md).

## Examples

The [MLX examples repo](https://github.com/ml-explore/mlx-examples) has a
variety of examples, including:

- [Transformer language model](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) training.
- Large-scale text generation with
  [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) and
  finetuning with [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora).
- Generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
- Speech recognition with [OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

## Quickstart

See the [quick start
guide](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
in the documentation.

## Installation

MLX is available on [PyPI](https://pypi.org/project/mlx/). To install MLX on
macOS, run:

```bash
pip install mlx
```

To install the CUDA backend on Linux, run:

```bash
pip install mlx[cuda]
```

To install a CPU-only Linux package, run:

```bash
pip install mlx[cpu]
```

Checkout the
[documentation](https://ml-explore.github.io/mlx/build/html/install.html#)
for more information on building the C++ and Python APIs from source.

## Contributing

Check out the [contribution guidelines](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](https://github.com/ml-explore/mlx/tree/main/ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX and wish to be acknowledged, please add your name to the list in your
pull request.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
