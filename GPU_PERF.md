# GPU performance work (`perf/gpu`)

This branch holds GPU changes for MLX and a benchmark to measure them. It is
based on `main` at `96fbf67`, the same upstream commit as `perf/cpu`, so it can
be reviewed and benchmarked on its own.

**No GPU numbers have been measured yet.** The branch was written on a Linux
machine without a GPU. Everything below about speed is a reason to expect a
change, not a measurement. The numbers will come from running the benchmark on
a Mac (Metal) or an NVIDIA GPU (CUDA).

## What changed

| Change | Backend | Checked | Run on a GPU |
|---|---|---|---|
| `benchmarks/python/gpu_perf_bench.py` | Metal and CUDA | Run end to end on the CPU with `--allow-cpu` | No |
| Softmax keeps a row in registers | CUDA | Compiled with nvcc 12.9 for sm_80, sm_89, sm_90a and sm_120 | No |
| LayerNorm keeps a row in registers | CUDA | Compiled with nvcc 12.9 for sm_80, sm_89, sm_90a and sm_120 | No |

There are **no Metal kernel changes**. The Metal softmax, layer norm and RMS
norm kernels already have a single-pass variant for rows up to 4096 elements
(`block_softmax`, `layer_norm_single_row`, `rms_single_row`), and Metal code
cannot be compiled on Linux, so this branch does not touch it. On a Mac,
`perf/gpu` and `main` should therefore time the same, and the run is useful as a
baseline and as a check that the benchmark works.

### CUDA softmax (`mlx/backend/cuda/softmax.cu`)

The kernel used to read every row from global memory twice: once to find the
maximum and the normalizer, and once more to write `exp(x - max) / sum`. The
host picks a block of up to 1024 threads that each read `16 / itemsize`
values, so any row of up to 4096 float32 or 8192 float16/bfloat16 values is
covered in one pass. In that case each thread now keeps the values it loaded
in registers and reuses them for the output. The longer-row path is
unchanged.

Expected effect: about one third less memory traffic (2 reads + 1 write
becomes 1 read + 1 write) for rows that fit, such as attention scores. The
gain will be smaller when the second read already hit L2. The results should
be bit-for-bit identical, because the arithmetic and its order did not change.

### CUDA LayerNorm (`mlx/backend/cuda/layer_norm.cu`, forward only)

The forward kernel read each row three times: for the mean, for the variance
and for the output. It now uses the same single-pass rule as softmax (up to
4096 float32 or 8192 float16/bfloat16 values per row, which covers most
transformer hidden sizes) and keeps the row in registers, so the row is read
once. Padding lanes are masked out of the variance sum, because `0 - mean` is
not zero. The terms are summed in the same order as before, so the output
should be identical. RMSNorm on CUDA already did this (`rms_norm_small`). The
VJP kernel is unchanged.

Expected effect: about half the memory traffic for x (3 reads + 1 write
becomes 1 read + 1 write, plus the small weight and bias reads).

Register use was compared with `cuobjdump --dump-resource-usage` on the sm_89
build. The softmax kernels use the same or fewer registers. The LayerNorm
kernels stay within the range they used before (40 to 56). No kernel gained
stack or spill space.

## The benchmark

`benchmarks/python/gpu_perf_bench.py` runs 56 cases on `mx.gpu` (49 with
`--quick`):

- matmul: 1024, 2048 and 4096 squared in float32, float16 and bfloat16, plus a
  batched GEMM and a GEMV
- softmax (including precise and a 32000-wide row) and logsumexp
- layer_norm and rms_norm at hidden sizes 768, 4096, 8192 and 16384
- exp, add, broadcast add, silu and where
- sum, max and argmax reductions
- sort and argsort
- transposed copies and astype
- conv2d
- `mx.fast.scaled_dot_product_attention` for prefill (causal, MHA and GQA) and
  decode

Each case is warmed up (5 calls by default) and then timed for 15 rounds. A
round is long enough to last at least 25 ms, and every call ends with
`mx.eval`. The JSON keeps the median and the minimum per call, every round,
a checksum of the output, the device info (`mx.device_info`, or
`mx.metal.device_info` on older builds, or `nvidia-smi`), the MLX version, and
the git commit, branch and dirty flag of the MLX tree that was imported.

Useful options:

- `--out FILE.json` writes the results.
- `--compare A.json B.json` prints per-op speedups (A over B, so above 1 means
  B is faster), a geomean per group and overall, and flags ops whose output
  checksum differs between the two runs.
- `--verify` also runs each case on the CPU and reports the largest relative
  difference. This checks that the kernels are correct, not just fast.
- `--quick` uses smaller shapes. Do not compare a quick run with a full one.
- `--filter REGEX` runs only the matching ops, and `--list` prints them.
- `--rounds`, `--warmup`, `--inner` and `--min-round-ms` control the timing.

## How to run it

Run `main` and `perf/gpu` on the same machine, plugged in, with other apps
closed. Copy the script out of the tree first, because `main` does not have
it.

### Mac (Apple silicon, Metal)

You need Xcode 15 or later (with the Metal toolchain), CMake 3.25 or later and
Python 3.10 or later, in a native arm64 shell (`uname -p` should print `arm`).

```shell
git clone https://github.com/shubhxho/mlx.git mlx-gpu && cd mlx-gpu
git checkout perf/gpu
cp benchmarks/python/gpu_perf_bench.py ~/gpu_perf_bench.py

python3 -m venv ~/mlx-venv && source ~/mlx-venv/bin/activate
pip install -U pip

# Baseline: main
git checkout main
pip install -e ".[dev]"
(cd ~ && python gpu_perf_bench.py --label main --out ~/metal_main.json)

# Candidate: perf/gpu
git checkout perf/gpu
python setup.py build_ext --inplace
(cd ~ && python gpu_perf_bench.py --label perf-gpu --out ~/metal_perf_gpu.json)

# Compare, and check correctness against the CPU
(cd ~ && python gpu_perf_bench.py --compare ~/metal_main.json ~/metal_perf_gpu.json)
(cd ~ && python gpu_perf_bench.py --quick --verify --filter "softmax|layer_norm|rms_norm|sdpa")
```

### Linux with an NVIDIA GPU (CUDA)

You need the CUDA toolkit 12.x (12.9 is what the docs use; 13.1 is not
supported), cuDNN 9, BLAS and LAPACK. On Ubuntu:

```shell
sudo apt-get install -y libblas-dev liblapack-dev liblapacke-dev libcudnn9-dev-cuda-12
# plus cuda-toolkit-12-9 from NVIDIA's apt repository (see docs/src/install.rst)
nvidia-smi   # confirm the GPU and driver
```

Then:

```shell
git clone https://github.com/shubhxho/mlx.git mlx-gpu && cd mlx-gpu
git checkout perf/gpu
cp benchmarks/python/gpu_perf_bench.py ~/gpu_perf_bench.py

python3 -m venv ~/mlx-venv && source ~/mlx-venv/bin/activate
pip install -U pip
export CMAKE_ARGS="-DMLX_BUILD_CUDA=ON"   # builds for the GPU in this machine

# Baseline: main (the first CUDA build takes a while)
git checkout main
pip install -e ".[dev]"
(cd ~ && python gpu_perf_bench.py --label main --out ~/cuda_main.json)

# Candidate: perf/gpu (only softmax.cu and layer_norm.cu need to rebuild)
git checkout perf/gpu
python setup.py build_ext --inplace
(cd ~ && python gpu_perf_bench.py --label perf-gpu --out ~/cuda_perf_gpu.json)

# Compare, and check correctness against the CPU
(cd ~ && python gpu_perf_bench.py --compare ~/cuda_main.json ~/cuda_perf_gpu.json)
(cd ~ && python gpu_perf_bench.py --quick --verify --filter "softmax|layer_norm|rms_norm|logsumexp")
```

If CMake cannot detect the GPU, add the architecture by hand, for example
`-DMLX_CUDA_ARCHITECTURES=89` for an RTX 40 series card or `120` for an
RTX 50 series card.

Also run the CUDA Python tests on `perf/gpu`, since the kernel changes have
only been compiled, not run:

```shell
python -m unittest discover python/tests
```

### What to send back

Send both JSON files, the `--compare` output and the `--verify` output. The
JSON records the commit of each run, so the results can be checked against the
right branch. Runs within about 3% of each other count as unchanged (change
this with `--threshold`). If a result looks surprising, run `main` again
afterwards to check the noise.

## Status

- CPU-only build of this branch: 243/243 C++ test cases and 775 Python tests
  pass (60 skipped). These match `main`; `perf/cpu` has 11 more Python tests.
- The CUDA changes compile with nvcc 12.9 but have not been run.
- No GPU speedups have been measured yet.
