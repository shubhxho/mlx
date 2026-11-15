# Copyright © 2026 Apple Inc.
"""GPU benchmark for MLX (Metal or CUDA).

Runs a fixed set of ops on ``mx.gpu`` and writes the timings to JSON, so two
builds (for example upstream ``main`` and ``perf/gpu``) can be compared on the
same machine.

Run one build::

    python benchmarks/python/gpu_perf_bench.py --out main.json

Compare two runs (A is the baseline, B is the candidate)::

    python benchmarks/python/gpu_perf_bench.py --compare main.json perf_gpu.json

Each case is warmed up, then timed for ``--rounds`` rounds of ``--inner``
calls. Every call is followed by ``mx.eval`` so the GPU work finishes before
the clock stops. The JSON keeps the median and minimum time per call, a
checksum of the output, and the device, MLX version and git commit.

``--verify`` also runs every case once on the CPU and reports the largest
difference, which checks GPU kernels for correctness, not just speed.
"""

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import mlx.core as mx

SCHEMA = 1

# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def _normal(shape, dtype, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(dtype)


def _case(name, group, make_inputs, fn, dtype):
    return {
        "name": name,
        "group": group,
        "make_inputs": make_inputs,
        "fn": fn,
        "dtype": dtype,
    }


def build_cases(quick=False):
    """Return the list of benchmark cases.

    ``quick`` shrinks the large shapes so a full run finishes in about a
    minute on small machines. Quick and full results should not be compared
    with each other, which is why the mode is recorded in the JSON.
    """
    s = 2048 if quick else 4096
    cases = []

    # Matmul: square GEMMs in each float dtype, a batched GEMM and a GEMV.
    for n in ([1024, 2048] if quick else [1024, 2048, 4096]):
        for dt in ["float32", "float16", "bfloat16"]:
            cases.append(
                _case(
                    f"matmul {n}x{n}x{n} {dt}",
                    "matmul",
                    lambda n=n, dt=dt: (
                        _normal((n, n), DTYPES[dt], 0),
                        _normal((n, n), DTYPES[dt], 1),
                    ),
                    lambda a, b: a @ b,
                    dt,
                )
            )
    cases.append(
        _case(
            "matmul batched 32x512x512x512 float16",
            "matmul",
            lambda: (
                _normal((32, 512, 512), mx.float16, 0),
                _normal((32, 512, 512), mx.float16, 1),
            ),
            lambda a, b: a @ b,
            "float16",
        )
    )
    cases.append(
        _case(
            f"gemv 1x{s} @ {s}x{s} float16",
            "matmul",
            lambda: (
                _normal((1, s), mx.float16, 0),
                _normal((s, s), mx.float16, 1),
            ),
            lambda a, b: a @ b,
            "float16",
        )
    )

    # Softmax: rows that fit in one pass, and a long (vocabulary sized) row.
    softmax_shapes = [
        ((s, s), "float32"),
        ((s, s), "float16"),
        ((8, 32, 512, 512) if not quick else (4, 16, 512, 512), "float16"),
        ((256 if quick else 1024, 32000), "float32"),
    ]
    for shape, dt in softmax_shapes:
        cases.append(
            _case(
                f"softmax {'x'.join(map(str, shape))} {dt}",
                "softmax",
                lambda shape=shape, dt=dt: (_normal(shape, DTYPES[dt], 0),),
                lambda x: mx.softmax(x, axis=-1),
                dt,
            )
        )
    cases.append(
        _case(
            f"softmax precise {s}x{s} float16",
            "softmax",
            lambda: (_normal((s, s), mx.float16, 0),),
            lambda x: mx.softmax(x, axis=-1, precise=True),
            "float16",
        )
    )
    cases.append(
        _case(
            f"logsumexp {s}x{s} float32",
            "softmax",
            lambda: (_normal((s, s), mx.float32, 0),),
            lambda x: mx.logsumexp(x, axis=-1),
            "float32",
        )
    )

    # Norms. Hidden sizes up to 1024 * (16 / itemsize) take the single pass
    # path on CUDA; 16384 in float32 takes the looped path.
    rows = 2048 if quick else 8192
    norm_shapes = [
        ((rows, 4096), "float32"),
        ((rows, 4096), "float16"),
        ((rows, 4096), "bfloat16"),
        ((rows // 2, 8192), "float16"),
        ((rows // 8, 16384), "float32"),
        ((rows, 768), "float32"),
    ]
    for shape, dt in norm_shapes:
        tag = f"{'x'.join(map(str, shape))} {dt}"
        d = shape[-1]
        cases.append(
            _case(
                f"layer_norm {tag}",
                "norm",
                lambda shape=shape, dt=dt, d=d: (
                    _normal(shape, DTYPES[dt], 0),
                    _normal((d,), DTYPES[dt], 1),
                    _normal((d,), DTYPES[dt], 2),
                ),
                lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
                dt,
            )
        )
        cases.append(
            _case(
                f"rms_norm {tag}",
                "norm",
                lambda shape=shape, dt=dt, d=d: (
                    _normal(shape, DTYPES[dt], 0),
                    _normal((d,), DTYPES[dt], 1),
                ),
                lambda x, w: mx.fast.rms_norm(x, w, 1e-5),
                dt,
            )
        )
    cases.append(
        _case(
            f"layer_norm no affine {rows}x4096 float16",
            "norm",
            lambda: (_normal((rows, 4096), mx.float16, 0),),
            lambda x: mx.fast.layer_norm(x, None, None, 1e-5),
            "float16",
        )
    )

    # Elementwise.
    for dt in ["float32", "float16"]:
        cases.append(
            _case(
                f"exp {s}x{s} {dt}",
                "elementwise",
                lambda dt=dt: (_normal((s, s), DTYPES[dt], 0),),
                mx.exp,
                dt,
            )
        )
    cases.append(
        _case(
            f"add {s}x{s} float32",
            "elementwise",
            lambda: (
                _normal((s, s), mx.float32, 0),
                _normal((s, s), mx.float32, 1),
            ),
            mx.add,
            "float32",
        )
    )
    cases.append(
        _case(
            f"add broadcast row {s}x{s} + {s} float32",
            "elementwise",
            lambda: (
                _normal((s, s), mx.float32, 0),
                _normal((s,), mx.float32, 1),
            ),
            mx.add,
            "float32",
        )
    )
    cases.append(
        _case(
            f"silu x*sigmoid(x) {s}x{s} float16",
            "elementwise",
            lambda: (_normal((s, s), mx.float16, 0),),
            lambda x: x * mx.sigmoid(x),
            "float16",
        )
    )
    cases.append(
        _case(
            f"where {s}x{s} float32",
            "elementwise",
            lambda: (
                _normal((s, s), mx.float32, 0) > 0,
                _normal((s, s), mx.float32, 1),
                _normal((s, s), mx.float32, 2),
            ),
            mx.where,
            "float32",
        )
    )

    # Reductions.
    red = [
        ("sum all", lambda x: mx.sum(x)),
        ("sum axis=-1", lambda x: mx.sum(x, axis=-1)),
        ("sum axis=0", lambda x: mx.sum(x, axis=0)),
        ("max axis=-1", lambda x: mx.max(x, axis=-1)),
        ("argmax axis=-1", lambda x: mx.argmax(x, axis=-1)),
    ]
    for label, fn in red:
        cases.append(
            _case(
                f"{label} {s}x{s} float32",
                "reduce",
                lambda: (_normal((s, s), mx.float32, 0),),
                fn,
                "float32",
            )
        )

    # Sort.
    sort_rows = 256 if quick else 1024
    cases.append(
        _case(
            f"sort axis=-1 {sort_rows}x4096 float32",
            "sort",
            lambda: (_normal((sort_rows, 4096), mx.float32, 0),),
            lambda x: mx.sort(x, axis=-1),
            "float32",
        )
    )
    cases.append(
        _case(
            f"argsort axis=-1 {sort_rows}x4096 float32",
            "sort",
            lambda: (_normal((sort_rows, 4096), mx.float32, 0),),
            lambda x: mx.argsort(x, axis=-1),
            "float32",
        )
    )
    cases.append(
        _case(
            f"sort axis=0 4096x{sort_rows} float32",
            "sort",
            lambda: (_normal((4096, sort_rows), mx.float32, 0),),
            lambda x: mx.sort(x, axis=0),
            "float32",
        )
    )

    # Copies.
    for dt in ["float32", "float16"]:
        cases.append(
            _case(
                f"transpose copy {s}x{s} {dt}",
                "copy",
                lambda dt=dt: (_normal((s, s), DTYPES[dt], 0),),
                lambda x: mx.contiguous(x.T),
                dt,
            )
        )
    cases.append(
        _case(
            "transpose copy 32x512x32x128 (0,2,1,3) float16",
            "copy",
            lambda: (_normal((32, 512, 32, 128), mx.float16, 0),),
            lambda x: mx.contiguous(mx.transpose(x, (0, 2, 1, 3))),
            "float16",
        )
    )
    cases.append(
        _case(
            f"astype float32->float16 {s}x{s}",
            "copy",
            lambda: (_normal((s, s), mx.float32, 0),),
            lambda x: x.astype(mx.float16),
            "float16",
        )
    )

    # Convolution (NHWC input, OHWI weights).
    conv_shapes = [
        ((8, 64, 64, 64), (128, 3, 3, 64), "float32"),
        ((8, 64, 64, 64), (128, 3, 3, 64), "float16"),
        ((16, 32, 32, 256), (256, 3, 3, 256), "float16"),
        ((8, 56, 56, 64), (256, 1, 1, 64), "float16"),
    ]
    if quick:
        conv_shapes = conv_shapes[:2]
    for xs, ws, dt in conv_shapes:
        cases.append(
            _case(
                f"conv2d x{'x'.join(map(str, xs))} w{'x'.join(map(str, ws))} {dt}",
                "conv",
                lambda xs=xs, ws=ws, dt=dt: (
                    _normal(xs, DTYPES[dt], 0),
                    _normal(ws, DTYPES[dt], 1)
                    * (1.0 / math.sqrt(ws[1] * ws[2] * ws[3])),
                ),
                lambda x, w: mx.conv2d(x, w, padding=w.shape[1] // 2),
                dt,
            )
        )

    # Attention.
    sdpa_shapes = [
        # (B, H_q, H_kv, L_q, L_kv, D, causal)
        (1, 32, 32, 2048, 2048, 128, True),
        (1, 32, 8, 2048, 2048, 128, True),
        (4, 16, 16, 512, 512, 64, False),
        (1, 32, 8, 1, 4096, 128, False),  # decode
    ]
    if quick:
        sdpa_shapes = [
            (1, 16, 16, 1024, 1024, 128, True),
            (1, 32, 8, 1, 2048, 128, False),
        ]
    for B, Hq, Hkv, Lq, Lkv, D, causal in sdpa_shapes:
        mask = "causal" if causal else None
        cases.append(
            _case(
                f"sdpa B{B} Hq{Hq} Hkv{Hkv} Lq{Lq} Lkv{Lkv} D{D}"
                f"{' causal' if causal else ''} float16",
                "attention",
                lambda B=B, Hq=Hq, Hkv=Hkv, Lq=Lq, Lkv=Lkv, D=D: (
                    _normal((B, Hq, Lq, D), mx.float16, 0),
                    _normal((B, Hkv, Lkv, D), mx.float16, 1),
                    _normal((B, Hkv, Lkv, D), mx.float16, 2),
                ),
                lambda q, k, v, D=D, mask=mask: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=1.0 / math.sqrt(D), mask=mask
                ),
                "float16",
            )
        )

    return cases


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _git_info():
    """Git commit and branch of the MLX source tree that was imported."""
    path = os.path.dirname(os.path.abspath(mx.__file__))
    info = {"mlx_path": path}
    for cmd, key in [
        (["git", "rev-parse", "HEAD"], "git_commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "git_branch"),
        (["git", "status", "--porcelain", "--untracked-files=no"], "git_dirty"),
    ]:
        try:
            out = subprocess.run(
                cmd, cwd=path, capture_output=True, text=True, timeout=10
            )
            if out.returncode == 0:
                val = out.stdout.strip()
                info[key] = bool(val) if key == "git_dirty" else val
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def device_info():
    info = {}
    backend = "unknown"
    if mx.metal.is_available():
        backend = "metal"
    elif hasattr(mx, "cuda") and mx.cuda.is_available():
        backend = "cuda"
    info["backend"] = backend
    raw = None
    try:
        raw = mx.device_info(mx.gpu)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        try:
            raw = mx.metal.device_info()
        except (AttributeError, RuntimeError):
            raw = None
    if raw:
        info.update({k: _jsonable(v) for k, v in raw.items()})
    if backend == "cuda" and "device_name" not in info:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                info["nvidia_smi"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def environment(args):
    env = {
        "schema": SCHEMA,
        "label": args.label,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": device_info(),
        "quick": args.quick,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "min_round_ms": args.min_round_ms,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    env.update(_git_info())
    return env


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def _checksum(out):
    """Sum and absolute sum of the output, in float32, as a fingerprint."""
    outs = out if isinstance(out, (list, tuple)) else [out]
    total = 0.0
    abs_total = 0.0
    for o in outs:
        o = o.astype(mx.float32)
        total += mx.sum(o).item()
        abs_total += mx.sum(mx.abs(o)).item()
    return [total, abs_total]


def _max_rel_diff(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    scale = mx.max(mx.abs(b)).item()
    diff = mx.max(mx.abs(a - b)).item()
    return diff / max(scale, 1e-30)


VERIFY_TOL = {"float32": 2e-3, "float16": 3e-2, "bfloat16": 6e-2}


def run_case(case, args):
    inputs = case["make_inputs"]()
    mx.eval(*inputs)
    fn = case["fn"]

    for _ in range(args.warmup):
        mx.eval(fn(*inputs))
    mx.synchronize()

    # Choose how many calls make up one round so a round lasts at least
    # min_round_ms, which keeps timer and launch noise small for fast ops.
    tic = time.perf_counter()
    mx.eval(fn(*inputs))
    single = time.perf_counter() - tic
    if args.inner:
        inner = args.inner
    else:
        inner = int(
            min(max(math.ceil(args.min_round_ms / 1e3 / max(single, 1e-7)), 1), 200)
        )

    per_call = []
    for _ in range(args.rounds):
        tic = time.perf_counter()
        for _ in range(inner):
            mx.eval(fn(*inputs))
        per_call.append((time.perf_counter() - tic) / inner)

    out = fn(*inputs)
    mx.eval(out)
    result = {
        "name": case["name"],
        "group": case["group"],
        "dtype": case["dtype"],
        "median_ms": 1e3 * statistics.median(per_call),
        "min_ms": 1e3 * min(per_call),
        "max_ms": 1e3 * max(per_call),
        "inner": inner,
        "rounds_ms": [1e3 * t for t in per_call],
        "checksum": _checksum(out),
    }

    if args.verify:
        # Integer outputs (argmax, argsort) can legitimately differ on ties,
        # so they are compared by the values they select instead.
        if case["group"] == "sort" and "argsort" in case["name"]:
            x = inputs[0]
            gpu_vals = mx.take_along_axis(x, out, axis=-1)
            with mx.stream(mx.cpu):
                cpu_vals = mx.sort(x, axis=-1)
                mx.eval(cpu_vals)
            rel = _max_rel_diff(gpu_vals, cpu_vals)
        elif "argmax" in case["name"]:
            x = inputs[0]
            gpu_vals = mx.take_along_axis(x, out[..., None], axis=-1)
            with mx.stream(mx.cpu):
                cpu_vals = mx.max(x, axis=-1, keepdims=True)
                mx.eval(cpu_vals)
            rel = _max_rel_diff(gpu_vals, cpu_vals)
        else:
            with mx.stream(mx.cpu):
                ref = fn(*inputs)
                mx.eval(ref)
            rel = _max_rel_diff(out, ref)
        tol = VERIFY_TOL.get(case["dtype"], 2e-3)
        # Long float32 reductions and matmuls accumulate in a different order
        # on each device, so give them more room.
        if case["group"] in ("matmul", "conv", "reduce", "attention"):
            tol *= 10
        result["verify_max_rel_diff"] = rel
        result["verify_ok"] = bool(rel <= tol)

    del inputs, out
    mx.clear_cache()
    return result


def run(args):
    cases = build_cases(quick=args.quick)
    if args.filter:
        pat = re.compile(args.filter)
        cases = [c for c in cases if pat.search(c["name"])]
    if args.list:
        for c in cases:
            print(c["name"])
        return
    if not cases:
        sys.exit("no ops match --filter")

    if not args.allow_cpu and not (
        mx.metal.is_available() or (hasattr(mx, "cuda") and mx.cuda.is_available())
    ):
        sys.exit(
            "No GPU backend found (mx.metal and mx.cuda are unavailable). "
            "Build MLX with Metal or CUDA, or pass --allow-cpu to time the CPU."
        )
    device = mx.cpu if args.allow_cpu and args.device == "cpu" else mx.gpu
    mx.set_default_device(device)

    env = environment(args)
    env["device_used"] = str(device)
    dev = env["device"]
    print(
        f"MLX {env['mlx_version']}  commit {env.get('git_commit', '?')[:12]}"
        f"  branch {env.get('git_branch', '?')}  backend {dev.get('backend')}"
        f"  device {dev.get('device_name', dev.get('architecture', '?'))}"
    )
    if env.get("git_dirty"):
        print("warning: the MLX source tree has uncommitted changes")

    results = []
    width = max(len(c["name"]) for c in cases)
    print(f"{'op':<{width}}  {'median ms':>10}  {'min ms':>10}  calls")
    for case in cases:
        try:
            r = run_case(case, args)
        except Exception as e:  # keep going so one bad op does not lose a run
            print(f"{case['name']:<{width}}  ERROR {type(e).__name__}: {e}")
            results.append(
                {"name": case["name"], "group": case["group"], "error": str(e)}
            )
            mx.clear_cache()
            continue
        line = f"{r['name']:<{width}}  {r['median_ms']:>10.4f}  {r['min_ms']:>10.4f}  {r['inner']}x{args.rounds}"
        if "verify_ok" in r:
            line += f"  verify {'ok' if r['verify_ok'] else 'MISMATCH'} ({r['verify_max_rel_diff']:.2e})"
        print(line, flush=True)
        results.append(r)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"env": env, "results": results}, f, indent=1)
        print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def _describe(env):
    dev = env.get("device", {})
    name = (
        dev.get("device_name")
        or dev.get("architecture")
        or dev.get("nvidia_smi")
        or "?"
    )
    return (
        f"{env.get('label') or ''} commit {str(env.get('git_commit', '?'))[:12]}"
        f" ({env.get('git_branch', '?')}{', dirty' if env.get('git_dirty') else ''})"
        f" MLX {env.get('mlx_version', '?')} {dev.get('backend', '?')} {name}"
    ).strip()


def compare(path_a, path_b, threshold):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    ea, eb = a["env"], b["env"]
    print(f"A (baseline):  {_describe(ea)}")
    print(f"B (candidate): {_describe(eb)}")
    da, db = ea.get("device", {}), eb.get("device", {})
    if da.get("device_name") != db.get("device_name") or da.get("backend") != db.get(
        "backend"
    ):
        print("warning: A and B ran on different devices; speedups are not meaningful")
    if ea.get("quick") != eb.get("quick"):
        print("warning: one run used --quick and the other did not; shapes differ")

    ra = {r["name"]: r for r in a["results"] if "median_ms" in r}
    rb = {r["name"]: r for r in b["results"] if "median_ms" in r}
    names = [r["name"] for r in a["results"] if r["name"] in ra and r["name"] in rb]
    if not names:
        sys.exit("no ops in common between the two files")

    width = max(len(n) for n in names)
    print()
    print(f"{'op':<{width}}  {'A ms':>10}  {'B ms':>10}  {'speedup':>8}  note")
    logs = []
    by_group = {}
    for n in names:
        ta, tb = ra[n]["median_ms"], rb[n]["median_ms"]
        sp = ta / tb if tb > 0 else float("nan")
        logs.append(math.log(sp))
        by_group.setdefault(ra[n]["group"], []).append(math.log(sp))
        notes = []
        if sp >= 1 + threshold:
            notes.append("faster")
        elif sp <= 1 / (1 + threshold):
            notes.append("SLOWER")
        ca, cb = ra[n].get("checksum"), rb[n].get("checksum")
        if ca and cb:
            scale = max(abs(ca[1]), abs(cb[1]), 1e-30)
            rel = max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) / scale
            if rel > 1e-2:
                notes.append(f"OUTPUT DIFFERS (checksum rel {rel:.1e})")
        for side, r in (("A", ra[n]), ("B", rb[n])):
            if r.get("verify_ok") is False:
                notes.append(f"{side} failed --verify")
        print(f"{n:<{width}}  {ta:>10.4f}  {tb:>10.4f}  {sp:>7.3f}x  {' '.join(notes)}")

    print()
    for g, ls in by_group.items():
        print(f"geomean {g:<12} {math.exp(sum(ls) / len(ls)):.3f}x over {len(ls)} ops")
    print(
        f"geomean {'all':<12} {math.exp(sum(logs) / len(logs)):.3f}x over {len(logs)} ops"
    )
    missing = sorted(set(ra) ^ set(rb))
    if missing:
        print(f"\nops in only one file: {', '.join(missing)}")
    errors = [r["name"] for r in a["results"] + b["results"] if "error" in r]
    if errors:
        print(f"ops that errored: {', '.join(sorted(set(errors)))}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", help="write results to this JSON file")
    p.add_argument("--label", default="", help="free-form label stored in the JSON")
    p.add_argument(
        "--rounds", type=int, default=15, help="timed rounds per op (default 15)"
    )
    p.add_argument(
        "--warmup", type=int, default=5, help="untimed warmup calls per op (default 5)"
    )
    p.add_argument(
        "--inner", type=int, default=0, help="calls per round (default: auto)"
    )
    p.add_argument(
        "--min-round-ms",
        type=float,
        default=25.0,
        help="target length of one round when --inner is auto (default 25)",
    )
    p.add_argument("--filter", help="only run ops whose name matches this regex")
    p.add_argument(
        "--quick", action="store_true", help="smaller shapes, for a fast smoke run"
    )
    p.add_argument(
        "--verify", action="store_true", help="also check every op against the CPU"
    )
    p.add_argument("--list", action="store_true", help="list the ops and exit")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow running without a GPU (for testing this script)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="compare two result files (A is the baseline)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="relative change below which compare calls an op unchanged (default 0.03)",
    )
    args = p.parse_args()

    if args.compare:
        compare(*args.compare, args.threshold)
    else:
        run(args)


if __name__ == "__main__":
    main()
